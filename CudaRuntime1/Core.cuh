#pragma once
#ifndef CORE_cuh

#define CORE_cuh
#include "Constants.cuh"
#include "AMath.cuh"
using namespace AMath;
namespace Cores {
	class Function {
	public:
		virtual double function(double x, double y) {
			return 0;
		};
		virtual double function(double x) {
			return 0;
		};
	};
	class Gauss : public Function {
		double rho = 1;
		double delta = C_delta;
	public:
		double function(double x) override {
			return rho * exp(-x * x / (delta * delta));
		}
	};
	class ZeroDest {
	public:
		double function(double x, double y) {
			return exp(-x * x) / sqrtPi;
		};
	};
	class Core2D {
	protected:
		double h = 0.2667;
		double ownX, ownY;
	public:
		__device__ Core2D() {
			h = 0.2667;
			ownX = ownY = 0.0;
		};
		__device__ Core2D(double h) {
			this->h = h;
			ownX = ownY = 0.0;
		};
		__host__ __device__ Core2D(double x, double y) {
			ownX = x;
			ownY = y;
		};
		__device__ double function(double x, double y) {

			/*double r = __dmul_rd(x, x);
			r += __dmul_rd(y, y);
			r = __dsqrt_rn(r);

			double ownR = __dmul_rd(ownX, ownX);
			ownR += __dmul_rd(ownY, ownY);
			ownR = __dsqrt_rn(ownR);

			double result = __dsqrt_rn(M_PI);
			result = __dmul_rd(h, result);
			result = __drcp_rd(result);
			result = __dmul_rd(result, result);
			result = __dmul_rd(result, exp(-__ddiv_rd(__dmul_rd(r, r),(__dmul_rd(h, h)))));*/

			double r = ownX - x;
			return coreFunction(r);
		};
		__device__ virtual double range(double x, double y) {
			return sqrt(x * x + y * y);
			//return __dsqrt_rn(__dadd_rd(__dmul_rd(x, x), __dmul_rd(y, y)));
		}
		__device__ virtual double coreFunction(double r) {
			return 0;
		};
	};
	class H_Core2D {
	protected:
		double h = 0.2667;
		double ownX, ownY;
	public:
		H_Core2D() {
			h = 0.2667;
			ownX = ownY = 0.0;
		};
		H_Core2D(double h) {
			this->h = h;
			ownX = ownY = 0.0;
		};
		H_Core2D(double x, double y) {
			ownX = x;
			ownY = y;
		};
		double function(double x, double y) {

			/*double r = __dmul_rd(x, x);
			r += __dmul_rd(y, y);
			r = __dsqrt_rn(r);

			double ownR = __dmul_rd(ownX, ownX);
			ownR += __dmul_rd(ownY, ownY);
			ownR = __dsqrt_rn(ownR);

			double result = __dsqrt_rn(M_PI);
			result = __dmul_rd(h, result);
			result = __drcp_rd(result);
			result = __dmul_rd(result, result);
			result = __dmul_rd(result, exp(-__ddiv_rd(__dmul_rd(r, r),(__dmul_rd(h, h)))));*/

			double r = ownX - x;
			return coreFunction(r);
		};
		virtual double range(double x, double y) {
			return sqrt(x * x + y * y);
			//return __dsqrt_rn(__dadd_rd(__dmul_rd(x, x), __dmul_rd(y, y)));
		}
		virtual double coreFunction(double r) {
			return 0;
		};
	};
	/*class CorePoly6 : public Core {
	public:
		double coreFunction(double r) override {
			return 315 * pow((h * h - r * r), 3) / (64 * M_PI * pow(h, 9));
		};
	};
	class CoreSpiky : public Core {
	public:
		double coreFunction(double r) override {
			return 15 * pow((h - r), 3) / (M_PI * pow(h, 6));
		}
	};
	class CoreViscosity : public Core {
	public:
		double coreFunction(double r) override {
			return 15 / (2 * M_PI * pow(h, 3)) - pow(r, 3) / (2 * pow(h, 3)) + pow(r, 2) / pow(h, 2) + h / (2 * r) - 1;
		}
	};*/

	class CoreFinal : public Core2D {
	public:
		__device__ CoreFinal() {}
		__device__ CoreFinal(double x, double y) {
			ownX = x;
			ownY = y;
		};
		__device__ double coreFunction(double r) override {
			return (1 / (sqrtPi * h)) * (1 / (sqrtPi * h)) * (1 / (sqrtPi * h)) * exp(-(r * r) / (h * h));
			/*double result = __dsqrt_rn(M_PI);
			result = __dmul_rd(h, result);
			result = __drcp_rd(result);
			result = __dmul_rd(result, result);
			result = __dmul_rd(result, exp(-__ddiv_rd(__dmul_rd(r, r), (__dmul_rd(h, h)))));
			return result;*/
		}
	};
	class H_CoreFinal : public H_Core2D {
	public:
		H_CoreFinal() {}
		H_CoreFinal(double x, double y) {
			h = 0.2667;
			ownX = x;
			ownY = y;
		};
		double coreFunction(double r) override {
			return (1 / (sqrtPi * h)) * (1 / (sqrtPi * h)) * (1 / (sqrtPi * h)) * exp(-(r * r) / (h * h));
			/*double result = __dsqrt_rn(M_PI);
			result = __dmul_rd(h, result);
			result = __drcp_rd(result);
			result = __dmul_rd(result, result);
			result = __dmul_rd(result, exp(-__ddiv_rd(__dmul_rd(r, r), (__dmul_rd(h, h)))));
			return result;*/
		}
	};
	class V {
	public:
		__device__ double function(double x, double y) {
			return x * x / 2;
		};
	};
	class H_V {
	public:
		double function(double x, double y) {
			return x * x / 2;
		};
	};
	class E {
		int alpha = 0;
		int s = 0;
	public:
		E() {};
		E(int alph, int es) {
			alpha = alph;
			s = es;
		}
		double function(double x) {
			return gamma0 * sqrt(1 + 4 * cos(x * dx / H) * cos(M_PI * s / M) + 4 * cos(M_PI * s / M) * cos(M_PI * s / M)) * cos(alpha * dx * x / H) * dx / (M_PI * H);
		};
	};

	class Delta {
	public:
		double func(int alpha, int s) {
			E e = E(alpha, s);
			Integral<E> integral = Integral<E>(e);
			return integral.integrate(-M_PI * H / dx, M_PI * H / dx);
		};
	};


	class DownFunction {
		Delta d;
		int alpha = 0;
		int s = 0;
	public:
		DownFunction() {};
		DownFunction(int alph, int S) {
			alpha = alph;
			s = S;
		};
		;		double function(double x) {
			return exp(sum(x));
		};
		double sum(double x) {
			double sum = 0;
			for (int a = 1; a <= a_length; a++)
			{
				if (T == 0)
					break;
				double temp = cos(a * x) * d.func(a, s) / (kB * T);
				sum += temp;
			}
			sum *= -1;
			return sum;
		};
	};
	class UPFunction {
		int alpha = 0;
		DownFunction DF;
	public:
		UPFunction() {};
		UPFunction(int alph, int s) {
			alpha = alph;
			DF = DownFunction(alph, s);
		};
		double function(double x) {
			return DF.function(x) * cos(alpha * x);
		};
	};
	class G {
	public:
		G() {};
		double function(int alpha, int s) {
			Delta d;
			Integral<UPFunction> integralUP = Integral<UPFunction>(UPFunction(alpha, s));
			Integral<DownFunction> integralDown = Integral<DownFunction>(DownFunction(alpha, s));
			return -alpha * d.func(alpha, s) / (gamma0) * (integralUP.integrate(-M_PI, M_PI) / integralDown.integrate(-M_PI, M_PI));
		};
	};
	class WavePart {
	public:
		double function() {
			return AlphaSum();
		}
		double LSum(int alpha) {
			double temp = 0;
			for (size_t l = 0; l <= l_length; l++)
			{
				Fractal fractal;
				Gamma gamma;
				temp = pow(alpha, 2 * l);
				temp /= fractal.function(l);
				temp /= pow(2, 2 * l);
				temp /= gamma.function(l + 2);
				if (l % 2 != 0) {
					temp *= -1;
				}
			}
			return temp;
		}
		double AlphaSum() {

			double temp = 0;
			for (size_t alpha = 1; alpha <= a_length; alpha++)
			{
				G G;
				for (size_t S = 1; S <= s_length; S++)
				{
					temp += alpha * LSum(alpha) * G.function(alpha, S);
				}

			}
			return temp;
		}
	};
}
#endif
