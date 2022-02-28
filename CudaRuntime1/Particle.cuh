#pragma once
#ifndef PARTICLE_cuh
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#define PARTICLE_cuh
namespace particle {

	class Fluid {
	public:

		float mass;
		double2* possition;
		double2* velocity;
		double2* pressure;
		double2* nextPossition;
		double2* nextVelocity;
		double2* nextPressure;

		bool calledCUDA;
		
		Fluid() {
			calledCUDA = false;
			mass = 1 / WIDTH * HEIGHT;
			allocateHost();
		}

		~Fluid()
		{
			if (calledCUDA) {
				cudaFree(nextPossition);
				cudaFree(nextVelocity);
			}
				
		}
			
		void allocateDevice() {
			calledCUDA = true;
			int size = sizeof(double2);
			size *= WIDTH * HEIGHT;
			cudaMalloc((void**)&possition, size);
			cudaMalloc((void**)&nextPossition, size);
			cudaMalloc((void**)&nextVelocity, size);
			cudaMalloc((void**)&nextPressure, size);
			cudaMalloc((void**)&velocity, size);
			cudaMalloc((void**)&pressure, size);
			cudaMalloc((void**)&velocity, size);
		}

		void allocateHost() {

			for (int i = 0; i < WIDTH; i++)
			{
				for (int j = 0; j < HEIGHT; j++)
				{
					/*possition[i][j] = make_float2(0.1 * i,0.1 * j);
					velocity[i][j] = make_float2(0,0);*/
				}
			}
		}
	};
}
#endif // PARTICLE_cuh