
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <list>
#include <thread>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <sm_60_atomic_functions.h>
#include <cuda_gl_interop.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "Core.cuh"
#include "AMath.cuh"
#include "Particle.cuh"
#include "Constants.cuh"

using namespace std;
using namespace thrust;
using namespace Cores;
using namespace AMath;
using namespace particle;

/* screen constants */

int step = 0;

__constant__ const double dev_mass = (1.0 / (WIDTH * HEIGHT));
__constant__ const double dt = 0.01;
const double Dt = 0.004;
__constant__ double hdt = 0.005;
__constant__ int* dev_h;
__constant__ int* dev_w;
__constant__ int* size_n;
__constant__ int dev_hCell;

int* dev_LCells;
double total_time = 0.0;
int* dev_CellsN;
__constant__ int* dev_CellSide;


double* dev_Pxx;
double* dev_Pxy;
double* dev_Pyy;
double* dev_density;

double* dev_densityDx;
double* dev_densityDy;
double* dev_densityDxx;
double* dev_densityDxy;
double* dev_densityDyy;
double* dev_Pressure;


double2* dev_a;
double2* dev_nextPossition;
double2* dev_nextVelocity;
double2* dev_mvelocity;
double2* dev_pvelocity;
double2* dev_possition;
double2* host_possition;
double2* dev_velocity;

double d_dx;
double d_dy;

host_vector<int> host_n;
host_vector<double> host_r;
int* device_n;
double* device_r;


/* OpenGL interoperability */
dim3 blocks, threads;
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;

/* charge selection */
const int detectChargeRange = 20;
int selectedChargeIndex = -1;
bool isDragging = false;

double h_zoom = 1;
double h_duy, h_dux = 0;



static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



void key(unsigned char key, int x, int y) {
	switch (key) {
	case 'z':
		h_zoom += 0.1;
		printf("%e\n", h_zoom);
		break;
	case 'x':
		h_zoom -= 0.1;
		printf("%e\n", h_zoom);
		break;
	case 'w':
		h_dux += 0.1;
		break;
	case 's':
		h_dux -= 0.1;
		break;
	case 'd':
		h_duy += 0.1;
		break;
	case 'a':
		h_duy -= 0.1;
		break;
	case 'r':
		h_duy = 0;
		h_dux = 0;
		h_zoom = 1;
		break;
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
		atomicAdd(a + px, (dev_mass) * (-1) * ((Pl + Pr) * dx - v.function(possition[px], 0) * dx / density[x])); //- gamma * u[px] * Wij / density[x]));
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
__global__ void calculatePressure(double* Pxx, double* Pxy, double* Pyy, double* density, double* dx, double* dy, double* dxdx, double* dxdy, double* dydy, double2* possition) {
	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	const int px = gridDim.x * blockIdx.y + blockIdx.x;
	if (x < (int)dev_w && px < (int)dev_w && x != px) {
		double2 pos = possition[px];
		CoreFinal core = CoreFinal(pos.x, pos.y);
		pos = possition[x];
		double part1 = (0.25 * dev_mass / density[x]);
		atomicAdd(Pxx + px, part1 * (dx[x] * dx[x] / density[x] - dxdx[x]) * core.function(pos.x, pos.y));
		atomicAdd(Pxy + px, part1 * (dx[x] * dy[x] / density[x] - dxdy[x]) * core.function(pos.x, pos.y));
		atomicAdd(Pyy + px, part1 * (dy[x] * dy[x] / density[x] - dydy[x]) * core.function(pos.x, pos.y));
	}
}
__global__ void initDensity(double* density, double* dx, double* dy, double* dxdx, double* dxdy, double* dydy, double2* possition) {


	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	const int px = gridDim.x * blockIdx.y + blockIdx.x;


	if (x < (int)dev_w && px < (int)dev_w && x != px) {
		double2 pos = possition[px];
		CoreFinal core = CoreFinal(pos.x, pos.y);
		Derivative<CoreFinal> W(core);
		pos = possition[x];
		atomicAdd(density + px, dev_mass * core.function(pos.x, pos.y));
		atomicAdd(dx + px, dev_mass * W.dx(pos.x, pos.y));
		atomicAdd(dy + px, dev_mass * W.dy(pos.x, pos.y));
		atomicAdd(dxdx + px, dev_mass * W.dxdx(pos.x, pos.y));
		atomicAdd(dydy + px, dev_mass * W.dydy(pos.x, pos.y));
		atomicAdd(dxdy + px, dev_mass * W.dxdy(pos.x, pos.y));
	}
}
__global__ void calculateDensity(double* density, double* dx, double* dy, double* dxdx, double* dxdy, double* dydy, double2* possition)
{

	const int x = blockDim.x * threadIdx.y + threadIdx.x;
	const int px = gridDim.x * blockIdx.y + blockIdx.x;
	__shared__ double2 temp[WIDTH][HEIGHT]; //pos[threadIdx.x][blockIdx.x]

	if (x < (int)dev_w && px < (int)dev_w && x != px) {
		double2 pos = possition[px];
		CoreFinal core = CoreFinal(pos.x, pos.y);
		Derivative<CoreFinal> W(core);
		pos = possition[x];
		atomicAdd(density + px, dev_mass * core.function(pos.x, pos.y));
		atomicAdd(dx + px, dev_mass * W.dx(pos.x, pos.y));
		atomicAdd(dy + px, dev_mass * W.dy(pos.x, pos.y));
		atomicAdd(dxdx + px, dev_mass * W.dxdx(pos.x, pos.y));
		atomicAdd(dydy + px, dev_mass * W.dydy(pos.x, pos.y));
		atomicAdd(dxdy + px, dev_mass * W.dxdy(pos.x, pos.y));
	}
}

__global__ void output(double* parametr, double* x) {
	printf("------------------------------\n");
	for (size_t i = 0; i < (int)dev_w; i++)
	{
		printf("%e,%e\n", x[i], parametr[i]);
	}
	printf("------------------------------\n");

}
__global__ void initPosition(double* pos) {
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	if (ix > (int)dev_w)
		return;
	pos[ix] = -2 + ix * 4 / 300.0;
}
__global__ void clearScreen(uchar4* screen) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	clear(screen[x]);
}
__global__ void initVel(double* matrix) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < (int)dev_w)
		matrix[idx] = 1.0;
}
__global__ void preparePos(double2* particle, double* r, int* n, unsigned int seed) {

	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < (int)dev_w * (int)dev_h) {
		if (x == 0) {
			particle[x].x = 0;
			particle[x].y = 0;
			return;
		}
		int xi = x;
		for (size_t i = 0; i < (int)size_n; i++)
		{
			if (xi - n[i] > 0) {
				xi -= n[i];
			}
			else {
				double dtPhi = 2 * M_PI / (n[i]);

				curandState_t state;
				curand_init(seed, /* the seed controls the sequence of random values that are produced */
					x, /* the sequence number is only important with multiple cores */
					0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&state);
				double esp = (double)(curand(&state) % 100) / 1000.0;
				particle[x].x = r[i] * cos((double)xi * dtPhi) + esp;
				esp = (double)(curand(&state) % 100) / 1000.0;
				particle[x].y = r[i] * sin((double)xi * dtPhi) + esp;

				return;
			}
		}
	}


}
__global__ void renderFrame(uchar4* screen, double2* particle, const double dx, const double dy, double zoom, double dux, double duy) {

	int n = blockIdx.x * blockDim.x + threadIdx.x;

	int x = ((particle[n].x + (dx) / (2 * zoom)) + dux) * (zoom * WIDTH / dx);
	int y = ((particle[n].y + (dy) / (2 * zoom)) + duy) * (zoom * HEIGHT / dy);

	int temp = x * WIDTH + y;
	if (x < WIDTH && y < HEIGHT && 
		x>0 && y> 0) {
		setColorParticle(screen[temp]);
	}

}
void Acceleration() {
	cudaMemset(dev_a, 0, WIDTH * HEIGHT * sizeof(double));
	///*calculateAcceleration << <dim3(128, 3, 1), dim3(128, 3, 1) >> > (
	//	dev_a,
	//	dev_velocity,
	//	dev_Pxx,
	//	dev_Pxy,
	//	dev_Pyy,
	//	dev_density,
	//	dev_possition);*/
}

void Pressure() {
	cudaMemset(dev_Pxx, 0, WIDTH * sizeof(double));
	cudaMemcpy(dev_Pxy, dev_Pxx, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_Pyy, dev_Pxx, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToDevice);
	calculatePressure << <dim3(128, 3, 1), dim3(128, 3, 1) >> > (dev_Pxx,
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

void calculateDensity() {

	cudaMemset(dev_density, 0, WIDTH * HEIGHT * sizeof(double));
	cudaMemcpy(dev_densityDy, dev_density, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_densityDx, dev_density, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_densityDxx, dev_density, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_densityDyy, dev_density, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_densityDxy, dev_density, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToDevice);

	calculateDensity << <HEIGHT, WIDTH >> > (
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
	/*calculateDensity();
	HANDLE_ERROR(cudaDeviceSynchronize());
	Pressure();
	HANDLE_ERROR(cudaDeviceSynchronize());
	Acceleration();
	HANDLE_ERROR(cudaDeviceSynchronize());
	Velosity << <128, 3 >> > (dev_velocity, dev_mvelocity, dev_pvelocity, dev_a, dev_possition);
	HANDLE_ERROR(cudaDeviceSynchronize());*/

}


int setOfN(int step, host_vector<double>& R, Gauss gauss, float Mp) {

	double alpha = F_alpha;
	double result = 0.0;
	double Hp = alpha;
	Hp *= (step == 0) ? sqrt(Mp / gauss.function(0)) : sqrt(Mp / gauss.function(R.back()));

	result = (2 * M_PI / Hp);
	Hp += (step != 0) ? R.back() : 0;
	R.push_back(Hp);
	return (int)result * R.back();
}
void prepare() {
	int N = WIDTH * HEIGHT;
	Gauss gauss = Gauss();
	Integral<Gauss> integer = Integral<Gauss>(gauss);
	float M = 2 * M_PI * integer.integrate(G_begin, G_end);
	float Mp = M / N;
	int D;
	int step = 0;

	while (N > 0) {
		D = setOfN(step, host_r, gauss, Mp);
		host_n.push_back(D);
		N -= D;
		step++;
	}
	step = host_r.size();

	cudaMalloc(&device_n, host_n.size() * sizeof(int));
	cudaMalloc(&device_r, host_n.size() * sizeof(double));
	cudaMemcpy(device_n, &host_n[0], host_n.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_r, &host_r[0], host_r.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(size_n, &step, sizeof(int));

	preparePos << <WIDTH, HEIGHT >> > (dev_possition, device_r, device_n, time(NULL));

	cudaFree(device_n);
	cudaFree(device_r);
}
void initalizate() {
	//Выделение памяти



	int size = sizeof(double2);
	size *= WIDTH * HEIGHT;

	cudaMalloc((void**)&dev_nextPossition, size);
	cudaMalloc((void**)&dev_nextVelocity, size);
	cudaMalloc((void**)&dev_mvelocity, size);
	cudaMalloc((void**)&dev_velocity, size);
	cudaMalloc((void**)&dev_possition, size);
	cudaMalloc((void**)&dev_pvelocity, size);
	cudaMalloc((void**)&dev_a, size);

	size = sizeof(double);
	size *= WIDTH * HEIGHT;

	cudaMalloc((void**)&dev_density, size);
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

	host_possition = (double2*)malloc(HEIGHT * WIDTH * sizeof(double2));

	prepare();

	HANDLE_ERROR(cudaDeviceSynchronize());
		
	calculateDensity();

	//initVel<<<3,128>>>(dev_velocity);
	//
	//initPosition <<<dim3(3, 1, 1), dim3(128, 1, 1) >> > (dev_possition);
	//
	//HANDLE_ERROR(cudaDeviceSynchronize());
	//	
	//calculateDensity();
	//HANDLE_ERROR(cudaDeviceSynchronize());
	//Pressure();
	//HANDLE_ERROR(cudaDeviceSynchronize());
	//Acceleration();
	//HANDLE_ERROR(cudaDeviceSynchronize());
	//initVelosity << <128, 3 >> > (dev_velocity, dev_mvelocity, dev_pvelocity, dev_a, dev_possition);
	//HANDLE_ERROR(cudaDeviceSynchronize());
}



void findAnchors(double2* pos) {

	double h_maxX = pos[0].x;
	double h_maxY = pos[0].y;
	double h_minY = pos[0].y;
	double h_minX = pos[0].x;

	for (size_t i = 0; i < WIDTH * HEIGHT; i++)
	{
		if (pos[i].x > h_maxX) {
			h_maxX = pos[i].x;
		}
		if (pos[i].x < h_minX) {
			h_minX = pos[i].x;
		}
		if (pos[i].y < h_minY) {
			h_minY = pos[i].y;
		}
		if (pos[i].y > h_maxY) {
			h_maxY = pos[i].y;
		}
	}

	h_maxX -= h_minX;
	h_maxY -= h_minY;

	d_dx = h_maxX;
	d_dy = h_maxY;

}

void idle(void) {

	if (step == 0) {
		initalizate();
	}

	cudaMemcpy(host_possition, dev_possition, HEIGHT * WIDTH * sizeof(double2), cudaMemcpyDeviceToHost);

	findAnchors(host_possition);

	uchar4* dev_screen;
	size_t size;




	HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &size, cuda_vbo_resource));

	// Render Image
	clearScreen << <HEIGHT, WIDTH >> > (dev_screen);
	HANDLE_ERROR(cudaDeviceSynchronize());

	renderFrame << <HEIGHT, WIDTH >> > (dev_screen, dev_possition, d_dx, d_dy, h_zoom, h_dux, h_duy);

	HANDLE_ERROR(cudaDeviceSynchronize());

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	glutPostRedisplay();

	step++;
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


void Trash() {
	////out << <1, 1 >> > ();
	///*UPFunction d(1);
	//for (double i = -M_PI; i < M_PI; i+= M_PI/100)
	//{
	//	printf("%e;%e\n", i, d.function(i));
	//}*/
	///*int alpha = 1;
	//Integral<UPFunction> integralUP = Integral<UPFunction>(UPFunction(alpha));
	//Integral<DownFunction> integralDown = Integral<DownFunction>(DownFunction(alpha));
	//printf("%e\n", (integralUP.integrate(-M_PI, M_PI) / integralDown.integrate(-M_PI, M_PI)));*/
	///*WavePart WP;
	//printf("%e", WP.function());*/

	//double Px[WIDTH];
	//double x[WIDTH];
	//double a[WIDTH];
	//double v[WIDTH];
	//double pv[WIDTH];
	//double mv[WIDTH];

	//double density[WIDTH];
	//double dx[WIDTH];
	//double m = 1.0 / WIDTH;
	//double currX = -4.0;
	//for (size_t i = 0; i < WIDTH; i++)
	//{
	//	x[i] = currX;
	//	v[i] = 0.0;
	//	a[i] = 0;
	//	Px[i] = 0;
	//	density[i] = 0;
	//	dx[i] = 0;
	//	currX += 8.0 / 300.0;
	//}
	//
	////denst

	//for (size_t i = 0; i < WIDTH; i++)
	//{
	//	for (size_t j = 0; j < WIDTH; j++)
	//	{
	//		
	//		H_CoreFinal core = H_CoreFinal(x[i], 0);
	//		H_Derivative<H_CoreFinal> W(core);
	//		if (i != j) {
	//			density[i] += m * core.function(x[j], 0);
	//			dx[i] += m * W.dx(x[j], 0);
	//		}
	//	}
	//	
	//}
	//for (size_t i = 0; i < WIDTH; i++)
	//{
	//	printf("%e,%e\n", x[i], density[i]);
	//}
	//printf("........\n");
	//getchar();
	/////pressure
	//for (size_t i = 0; i < WIDTH; i++)
	//{
	//	for (size_t j = 0; j < WIDTH; j++)
	//	{
	//		H_CoreFinal core = H_CoreFinal(x[i], 0);
	//		H_Derivative<H_CoreFinal> W(core);
	//		double part1 = (0.25 * m / density[j]);
	//		if (i != j) {
	//			Px[i] += (dx[j] / density[j] - dx[j]) * core.function(x[j], 0);
	//		}
	//	}

	//}
	////accel
	//for (size_t i = 0; i < WIDTH; i++)
	//{
	//	for (size_t j = 0; j < WIDTH; j++)
	//	{
	//		H_CoreFinal core = H_CoreFinal(x[i], 0);
	//		H_Operator<H_CoreFinal> W = H_Operator<H_CoreFinal>(core);
	//		H_V V;
	//		double Pr = Px[j] / (density[j] * density[j]);
	//		double Pl = Px[i] / (density[i] * density[i]);
	//		double dx = W.Nabla(x[j], 0);
	//		double Wij = core.function(x[j], 0);
	//		if (i != j) {
	//			
	//			a[i] += (m) * ((-1) * (Pl + Pr) * dx - V.function(x[i], 0) * dx / density[j]  );
	//		}
	//		a[i] -= 4.0 * v[i];
	//	}

	//}

	////vel
	//
	//for (size_t i = 0; i < WIDTH; i++)
	//{
	//	mv[i] = v[i] - 0.5 * Dt * a[i];
	//	pv[i] = mv[i] + Dt * a[i];
	//	x[i] += Dt * pv[i];
	//	v[i] = 0.5 * (mv[i] + pv[i]);
	//	mv[i] = pv[i];
	//}

	//while (true) {
	//	
	//	for (size_t i = 0; i < WIDTH; i++)
	//	{
	//		a[i] = 0;
	//		Px[i] = 0;
	//		density[i] = 0;
	//		dx[i] = 0;
	//	}
	//	//denst

	//	for (size_t i = 0; i < WIDTH; i++)
	//	{
	//		
	//		for (size_t j = 0; j < WIDTH; j++)
	//		{
	//			H_CoreFinal core = H_CoreFinal(x[i], 0);
	//			H_Derivative<H_CoreFinal> W(core);
	//			if (i != j) {
	//				density[i] += m * core.function(x[j], 0);
	//				dx[i] += m * W.dx(x[j], 0);
	//			}
	//		}

	//	}
	//	for (size_t i = 0; i < WIDTH; i++)
	//	{
	//		printf("%e,%e\n", x[i], density[i]);
	//	}
	//	printf("........\n");
	//	getchar();
	//	///pressure
	//	for (size_t i = 0; i < WIDTH; i++)
	//	{
	//		for (size_t j = 0; j < WIDTH; j++)
	//		{
	//			H_CoreFinal core = H_CoreFinal(x[i], 0);
	//			H_Derivative<H_CoreFinal> W(core);
	//			double part1 = (0.25 * m / density[j]);
	//			if (i != j) {
	//				Px[i] += (dx[j] / density[j] - dx[j]) * core.function(x[j], 0);
	//			}
	//			
	//		}

	//	}
	//	//accel
	//	for (size_t i = 0; i < WIDTH; i++)
	//	{
	//		for (size_t j = 0; j < WIDTH; j++)
	//		{
	//			H_CoreFinal core = H_CoreFinal(x[i], 0);
	//			H_Operator<H_CoreFinal> W = H_Operator<H_CoreFinal>(core);
	//			H_V v;
	//			double Pr = Px[j] / (density[j] * density[j]);
	//			double Pl = Px[i] / (density[i] * density[i]);
	//			double dx = W.Nabla(x[j], 0);
	//			double Wij = core.function(x[j], 0);
	//			if (i != j) {
	//				a[i] += (m) * ((-1) * (Pl + Pr) * dx - v.function(x[i], 0) * dx / density[j]);
	//			}
	//		}

	//	}

	//	//vel

	//	for (size_t i = 0; i < WIDTH; i++)
	//	{
	//		pv[i] = mv[i] + Dt * a[i];
	//		x[i] += Dt * pv[i];
	//		v[i] = 0.5 * (mv[i] + pv[i]);
	//		mv[i] = pv[i];
	//	}
	//}




}

__global__ void temp(double2* pos) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%e;%e\n", pos[x].x, pos[x].y);
}

int main(int argc, char** argv) {



	setbuf(stdout, NULL);

	initCuda(0);

	initGlut(argc, argv);

	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	glutMainLoop();

	deleteVBO(&vbo, cuda_vbo_resource);



	//temp << <20, 20 >> > (dev_possition);



	freeMem();


	return 0;
}
__interface ICountable
{

};
class PoolTreads {
	list<thread> freeTreads;
	list<thread> occupiedTreads;
public:
	thread get() {
		thread worker;


		return worker;
	}
};

