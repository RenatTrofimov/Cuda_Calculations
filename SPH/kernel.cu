
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <sm_60_atomic_functions.h>
#include <cuda_gl_interop.h>


#include "Core.cuh"
#include "AMath.cuh"
#include "Particle.cuh"
#include "Constants.cuh"

using namespace std;
using namespace Cores;
using namespace AMath;
using namespace particle;

/* screen constants */


__constant__ const double dev_mass = (1.0 / (WIDTH));
__constant__ const double dt = 0.01;
__constant__ double hdt = 0.005;
__constant__ int* dev_h;
__constant__ int* dev_w;
__constant__ int dev_hCell;

int* dev_LCells;
double total_time = 0.0;
int* dev_CellsN;
__constant__ int* dev_CellSide;

double* dev_a;
double* dev_Pxx;
double* dev_Pxy;
double* dev_Pyy;
double* dev_density;
double* dev_W;
double* dev_densityDx;
double* dev_densityDy;
double* dev_densityDxx;
double* dev_densityDxy;
double* dev_densityDyy;
double* dev_Pressure;
double* dev_possition;
double* dev_velocity;
double* dev_mvelocity;
double* dev_pvelocity;
double2* dev_nextPossition;
double2* dev_nextVelocity;

/* OpenGL interoperability */
dim3 blocks, threads;
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;

/* charge selection */
const int detectChargeRange = 20;
int selectedChargeIndex = -1;
bool isDragging = false;



static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



void key(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		printf("Exit application\n");

		glutLeaveMainLoop();
		break;
	}
}

__device__ void setColorParticle(uchar4& pixel) {
	pixel.x = (0);
	pixel.y = (255);
	pixel.z = (0);
	pixel.w = (10);
}
__device__ void clear(uchar4& pixel) {
	pixel.x = (0);
	pixel.y = (0);
	pixel.z = (0);
	pixel.w = (0);
}
__global__ void calculateAcceleration(double* a, double* u, double* Pxx, double* Pxy, double* Pyy, double* density, double* possition) {
	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	const int px = gridDim.x * blockIdx.y + blockIdx.x;
	if (x < (int)dev_w && px < (int)dev_w && x != px) {
		double pos = possition[px];
		CoreFinal core = CoreFinal(pos, 0);
		//Derivative<CoreFinal> W(core);
		Operator<CoreFinal> W = Operator<CoreFinal>(core);
		V v;
		double Pl = Pxx[px] / (density[px] * density[px]);
		double Pr = Pxx[x] / (density[x] * density[x]);
		pos = possition[x];
		double dx = W.Nabla(pos, 0);
		double Wij = core.function(pos, 0);
		atomicAdd(a + px, (-1) * (dev_mass) * ((Pl + Pr) * dx - v.function(possition[px], 0) * dx / density[x])); //- gamma * u[px] * Wij / density[x]));
	}
}
__global__ void initVelosity(double* dev_velocity,
	double* dev_mvelocity,
	double* dev_pvelocity,
	double* dev_a,
	double* pos) {
	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	if (x < (int)dev_w) {
		dev_mvelocity[x] = dev_velocity[x] - 0.5 * dt * dev_a[x];
		dev_pvelocity[x] = dev_mvelocity[x] + dt * dev_a[x];
		pos[x] = pos[x] + dt * dev_pvelocity[x];
		pos[x];
		dev_velocity[x] = 0.5 * (dev_mvelocity[x] + dev_pvelocity[x]);
		dev_mvelocity[x] = dev_pvelocity[x];
	}
}
__global__ void Velosity(double* dev_velocity,
	double* dev_mvelocity,
	double* dev_pvelocity,
	double* dev_a,
	double* pos) {
	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	if (x < (int)dev_w) {
		dev_pvelocity[x] = dev_mvelocity[x] + dt * dev_a[x];
		pos[x] = pos[x] + dt * dev_pvelocity[x];
		dev_velocity[x] = 0.5 * (dev_mvelocity[x] + dev_pvelocity[x]);
		dev_mvelocity[x] = dev_pvelocity[x];
	}
}
__global__ void calcVelosity(double* dev_velocity,
	double* dev_mvelocity,
	double* dev_pvelocity,
	double* dev_a,
	double* pos) {
	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	if (x < (int)dev_w) {
		dev_pvelocity[x] = dev_mvelocity[x] + dt * dev_a[x];
		pos[x] = pos[x] + dt * dev_pvelocity[x];
		dev_velocity[x] = 0.5 * (dev_mvelocity[x] + dev_pvelocity[x]);
		dev_mvelocity[x] = dev_pvelocity[x];
	}
}
__global__ void calculatePressure(double* Pxx, double* Pxy, double* Pyy, double* density, double* dx, double* dy, double* dxdx, double* dxdy, double* dydy, double* possition) {
	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	const int px = gridDim.x * blockIdx.y + blockIdx.x;
	if (x < (int)dev_w && px < (int)dev_w && x != px) {
		double pos = possition[px];
		CoreFinal core = CoreFinal(pos, 0);
		pos = possition[x];
		double part1 = (0.25 * dev_mass / density[x]);
		atomicAdd(Pxx + px, density[px] * density[px] + part1 * (dx[x] * dx[x] / density[x] - dxdx[x]) * core.function(pos, 0));
		atomicAdd(Pxy + px, density[px] * density[px] + part1 * (dx[x] * dy[x] / density[x] - dxdy[x]) * core.function(pos, 0));
		atomicAdd(Pyy + px, density[px] * density[px] + part1 * (dy[x] * dy[x] / density[x] - dydy[x]) * core.function(pos, 0));
	}
}
__global__ void initDensity(double* density, double* dx, double* dy, double* dxdx, double* dxdy, double* dydy, double* possition){
	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	const int px = gridDim.x * blockIdx.y + blockIdx.x;
	if (x < (int)dev_w && px < (int)dev_w && x != px) {
		double pos = possition[px];
		CoreFinal core = CoreFinal(pos, 0);
		Derivative<CoreFinal> W(core);
		pos = possition[x];
		density[px] = exp(-(possition[px]) * (possition[px])) / sqrtPi;
		atomicAdd(dx + px, dev_mass * W.dx(pos, 0));
		atomicAdd(dy + px, dev_mass * W.dy(pos, 0));
		atomicAdd(dxdx + px, dev_mass * W.dxdx(pos, 0));
		atomicAdd(dydy + px, dev_mass * W.dydy(pos, 0));
		atomicAdd(dxdy + px, dev_mass * W.dxdy(pos, 0));
	}
}
__global__ void calculateDensity(double* density, double* dx, double* dy, double* dxdx, double* dxdy, double* dydy, double* possition)
{

	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	const int px = gridDim.x * blockIdx.y + blockIdx.x;


	if (x < (int)dev_w && px < (int)dev_w && x != px) {
		double pos = possition[px];
		CoreFinal core = CoreFinal(pos, 0);
		Derivative<CoreFinal> W(core);
		pos = possition[x];
		//density[px] = exp(-(possition[px]) * (possition[px])) / sqrtPi;
		atomicAdd(density + px, dev_mass * core.function(pos, 0));
		atomicAdd(dx + px, dev_mass * W.dx(pos, 0));
		atomicAdd(dy + px, dev_mass * W.dy(pos, 0));
		atomicAdd(dxdx + px, dev_mass * W.dxdx(pos, 0));
		atomicAdd(dydy + px, dev_mass * W.dydy(pos, 0));
		atomicAdd(dxdy + px, dev_mass * W.dxdy(pos, 0));
	}
}

__global__ void output(double* ro, double* x) {
	printf("------------------------------\n");
	for (size_t i = 0; i < (int)dev_w; i++)
	{
		printf("%e,%e\n", x[i], ro[i]);
	}
	printf("------------------------------\n");

}
__global__ void initPosition(double* pos) {
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	if (ix > (int)dev_w)
		return;
	pos[ix] = -2 + ix * 4 / 300.0;
}
__global__ void renderFrame(uchar4* screen, float2* grid) {

	grid[blockIdx.x * HEIGHT + threadIdx.x].x += 0.5;
	grid[blockIdx.x * HEIGHT + threadIdx.x].y += 0.5;
	clear(screen[blockIdx.x * HEIGHT + threadIdx.x]);

	int x = grid[blockIdx.x * HEIGHT + threadIdx.x].x;
	int y = grid[blockIdx.x * HEIGHT + threadIdx.x].y;

	setColorParticle(screen[x * HEIGHT + y]);
}
void Acceleration() {
	cudaMemset(dev_a, 0, WIDTH * sizeof(double));
	calculateAcceleration << <dim3(128, 3, 1), dim3(128, 3, 1) >> > (
		dev_a,
		dev_velocity,
		dev_Pxx,
		dev_Pxy,
		dev_Pyy,
		dev_density,
		dev_possition);
}

void Pressure(){
	cudaMemset(dev_Pxx, 0, WIDTH* sizeof(double));
	cudaMemcpy(dev_Pxy, dev_Pxx, WIDTH * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_Pyy, dev_Pxx, WIDTH * sizeof(double), cudaMemcpyDeviceToDevice);
	calculatePressure <<<dim3(128, 3, 1), dim3(128, 3, 1) >>> (dev_Pxx,
		dev_Pxy,
		dev_Pyy, 
		dev_density,
		dev_densityDx,
		dev_densityDy,
		dev_densityDxx,
		dev_densityDxy,
		dev_densityDyy,
		dev_possition);
}

void calculateDensity(){
	
	cudaMemset(dev_density, 0, WIDTH * sizeof(double));
	cudaMemcpy(dev_densityDy, dev_W, WIDTH * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_densityDx, dev_W, WIDTH * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_densityDxx, dev_W, WIDTH * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_densityDyy, dev_W, WIDTH * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_densityDxy, dev_W, WIDTH * sizeof(double), cudaMemcpyDeviceToDevice);

	calculateDensity << <dim3(128, 3, 1), dim3(128, 3, 1) >> > (
		dev_density,
		dev_densityDx,
		dev_densityDy,
		dev_densityDxx,
		dev_densityDxy,
		dev_densityDyy,
		dev_possition);
}

void freeMem() {
	cudaFree(dev_a);
	cudaFree(dev_velocity);
	cudaFree(dev_mvelocity);
	cudaFree(dev_pvelocity);
	cudaFree(dev_possition);
	cudaFree(dev_nextPossition);
	cudaFree(dev_nextVelocity);
	cudaFree(dev_density);
	cudaFree(dev_W);
	cudaFree(dev_densityDx);
	cudaFree(dev_densityDy);
	cudaFree(dev_densityDxy);
	cudaFree(dev_densityDxx);
	cudaFree(dev_densityDyy);
	cudaFree(dev_Pxx);
	cudaFree(dev_Pyy);
	cudaFree(dev_Pxy);
	cudaFree(dev_Pressure);
}
void calculation() {
	calculateDensity();
	HANDLE_ERROR(cudaDeviceSynchronize());
	Pressure();
	HANDLE_ERROR(cudaDeviceSynchronize());
	Acceleration();
	HANDLE_ERROR(cudaDeviceSynchronize());
	Velosity << <128, 3 >> > (dev_velocity, dev_mvelocity, dev_pvelocity, dev_a, dev_possition);
	HANDLE_ERROR(cudaDeviceSynchronize());
}
__global__ void initVel(double* matrix) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < (int)dev_w)
		matrix[idx] = 1.0;
}
void initalizate() {
	//Выделение памяти
	int size = sizeof(double2);
	size *= WIDTH;
	
	cudaMalloc((void**)&dev_nextPossition, size);
	cudaMalloc((void**)&dev_nextVelocity, size);

	size = sizeof(double);
	size *= WIDTH;
	cudaMalloc((void**)&dev_velocity, size);
	cudaMalloc((void**)&dev_mvelocity, size);
	cudaMalloc((void**)&dev_pvelocity, size);
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_possition, size);
	cudaMalloc((void**)&dev_density, size);
	cudaMalloc((void**)&dev_W, size);
	cudaMalloc((void**)&dev_densityDx, size);
	cudaMalloc((void**)&dev_densityDy, size);
	cudaMalloc((void**)&dev_densityDxy, size);
	cudaMalloc((void**)&dev_densityDxx, size);
	cudaMalloc((void**)&dev_densityDyy, size);
	cudaMalloc((void**)&dev_Pxx, size);
	cudaMalloc((void**)&dev_Pxy, size);
	cudaMalloc((void**)&dev_Pyy, size);
	cudaMalloc((void**)&dev_Pressure, size);
	
	int width = WIDTH;
	int height = HEIGHT;
	
	cudaMemcpyToSymbol(dev_w, &width, sizeof(int));
	cudaMemcpyToSymbol(dev_h, &height, sizeof(int));
	
	initVel<<<3,128>>>(dev_velocity);
	
	initPosition << <dim3(3, 1, 1), dim3(128, 1, 1) >> > (dev_possition);
	
	HANDLE_ERROR(cudaDeviceSynchronize());
	//Вычисления первого шага
	calculateDensity();
	HANDLE_ERROR(cudaDeviceSynchronize());
	Pressure();
	HANDLE_ERROR(cudaDeviceSynchronize());
	Acceleration();
	HANDLE_ERROR(cudaDeviceSynchronize());
	initVelosity << <128, 3 >> > (dev_velocity, dev_mvelocity, dev_pvelocity, dev_a, dev_possition);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void idle(void) {
	uchar4* dev_screen;
	size_t size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &size, cuda_vbo_resource));
	
	calculation();
	
	//Render Image
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	glutPostRedisplay();
}

void draw(void) {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glutSwapBuffers();
}

void createVBO(GLuint* vbo, 
	struct cudaGraphicsResource** vbo_res,
	unsigned int vbo_res_flags)
{
	unsigned int size = WIDTH * HEIGHT * sizeof(uchar4);

	glGenBuffers(1, vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res) {
	HANDLE_ERROR(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

void initCuda(int deviceId) {
	int deviceCount = 0;
	HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

	if (deviceCount <= 0) {
		printf("No CUDA devices found\n");
		exit(-1);
	}

	HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceProperties(&properties, deviceId));

	threads.x = 128;
	threads.y = properties.maxThreadsPerBlock / threads.x; // to avoid cudaErrorLaunchOutOfResources error

	blocks.x = (WIDTH + threads.x - 1) / threads.x;
	blocks.y = (HEIGHT + threads.y - 1) / threads.y;

	printf(
		"Debug: blocks(%d, %d), threads(%d, %d)\nCalculated Resolution: %d x %d\n",
		blocks.x, blocks.y, threads.x, threads.y, blocks.x * threads.x,
		blocks.y * threads.y);
	
}
void initGlut(int argc, char** argv) {
	// Initialize freeglut
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("SPH");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

	glutIdleFunc(idle);
	glutKeyboardFunc(key);

	glutDisplayFunc(draw);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)WIDTH, 0.0, (GLdouble)HEIGHT);

	glewInit();
}
__global__ void out() {
	CoreFinal core = CoreFinal(0, 0);
	Derivative<CoreFinal> W = Derivative<CoreFinal>(core);
	for (double i = -2; i <= 2; i += 0.01)
	{
		printf("%e;%e\n", i, W.dxdx(i, 0));
	}
	printf("Begin\n");
}
int main(int argc, char** argv) {
	
	//out << <1, 1 >> > ();
	/*UPFunction d(1);
	for (double i = -M_PI; i < M_PI; i+= M_PI/100)
	{
		printf("%e;%e\n", i, d.function(i));
	}*/
	/*int alpha = 1;
	Integral<UPFunction> integralUP = Integral<UPFunction>(UPFunction(alpha));
	Integral<DownFunction> integralDown = Integral<DownFunction>(DownFunction(alpha));
	printf("%e\n", (integralUP.integrate(-M_PI, M_PI) / integralDown.integrate(-M_PI, M_PI)));*/
	/*WavePart WP;
	printf("%e", WP.function());*/
	
	
	setbuf(stdout, NULL);

	initCuda(0);
	
	initGlut(argc, argv);
	
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
	initalizate();
	glutMainLoop();

	deleteVBO(&vbo, cuda_vbo_resource);
	freeMem();
	return 0;
}