#pragma once
#ifndef AMath_cuh

#define AMath_cuh
namespace AMath {

	template<class T> class H_Derivative {
		double h;
		T function;
	public:
		H_Derivative() {
			function = T();
			h = 1e-5;
		};
		H_Derivative(T func) {
			function = func;
			h = 1e-5;
		};
		H_Derivative(double anoter_h) {
			function = T();
			h = anoter_h;
		};
		H_Derivative(double anoter_h, T func) {
			function = func;
			h = anoter_h;
		};
		double dx(double x, double y) {
			//return __ddiv_rd(__dsub_rd(function.function(__dadd_rd(x, h), y), function.function(x, y)), h);
			return (function.function(x + h, y) - function.function(x, y)) / h;
		};
		double dxdx(double x, double y) {
			//return __ddiv_rd(__dsub_rd(function.function(x, __dadd_rd(y, h)), function.function(x, y)), h);
			return (function.function(x + 2 * h, y) - 2 * function.function(x + h, y) + function.function(x, y)) / (h * h);
		};
		double dy(double x, double y) {
			//return __ddiv_rd(__dsub_rd(function.function(x, __dadd_rd(y, h)), function.function(x, y)), h);
			return (function.function(x, y + h) - function.function(x, y)) / h;
		};
		double dydy(double x, double y) {
			return (function.function(x, y + 2 * h) - 2 * function.function(x, y + h) + function.function(x, y)) / (h * h);
		};
		double dxdy(double x, double y) {
			return (
				function.function(x + h, y + h)
				- function.function(x + h, y)
				- function.function(x, y + h)
				+ function.function(x, y))
				/ (h * h);
		};
	};
	template<class T> class Derivative {
		double h;
		T function;
	public:
		__host__ __device__ Derivative() {
			function = T();
			h = 1e-5;
		};
		__host__ __device__ Derivative(T func) {
			function = func;
			h = 1e-5;
		};
		__host__ __device__ Derivative(double anoter_h) {
			function = T();
			h = anoter_h;
		};
		__host__ __device__ Derivative(double anoter_h, T func) {
			function = func;
			h = anoter_h;
		};
		__device__ double dx(double x, double y) {
			//return __ddiv_rd(__dsub_rd(function.function(__dadd_rd(x, h), y), function.function(x, y)), h);
			return (function.function(x + h, y) - function.function(x, y)) / h;
		};
		__device__ double dxdx(double x, double y) {
			//return __ddiv_rd(__dsub_rd(function.function(x, __dadd_rd(y, h)), function.function(x, y)), h);
			return (function.function(x + 2 * h, y) - 2 * function.function(x + h, y) + function.function(x, y)) / (h * h);
		};
		__device__ double dy(double x, double y) {
			//return __ddiv_rd(__dsub_rd(function.function(x, __dadd_rd(y, h)), function.function(x, y)), h);
			return (function.function(x, y + h) - function.function(x, y)) / h;
		};
		__device__ double dydy(double x, double y) {
			return (function.function(x, y + 2 * h) - 2 * function.function(x, y + h) + function.function(x, y)) / (h * h);
		};
		__device__ double dxdy(double x, double y) {
			return (
				function.function(x + h, y + h) 
				- function.function(x + h, y) 
				- function.function(x, y + h) 
				+ function.function(x, y)) 
				/ (h * h);
		};
	};
	template<class T> class H_Operator {
		H_Derivative<T> f;
	public:
		H_Operator() {
			f = H_Derivative<T>();
		};
		H_Operator(T t) {
			f = H_Derivative<T>(t);
		};
		double Laplas(double x, double y) {
			return f.dxdx(x, y) + f.dydy(x, y);
		};
		double Nabla(double x, double y) {
			return sqrt(f.dx(x, y) * f.dx(x, y) + f.dy(x, y) * f.dy(x, y));
		};
		double Div(double x, double y) {
			return sqrt(f.dx(x, y) * f.dx(x, y) + f.dy(x, y) * f.dy(x, y));
		};
	};
	template<class T> class Operator {
		Derivative<T> f;
	public:
		__host__ __device__ Operator() {
			f = Derivative<T>();
		};
		__host__ __device__ Operator(T t) {
			f = Derivative<T>(t);
		};
		__host__ __device__ double Laplas(double x, double y) {
			return f.dxdx(x, y) + f.dydy(x, y);
		};
		__host__ __device__ double Nabla(double x, double y) {
			return sqrt(f.dx(x, y) * f.dx(x, y) + f.dy(x, y) * f.dy(x, y));
		};
		__host__ __device__ double Div(double x, double y) {
			return sqrt(f.dx(x, y) * f.dx(x, y) + f.dy(x, y) * f.dy(x, y));
		};
	};
	template<class T> class Integral {
		double h;
		T function;
	public:
		Integral(T func) {
			h = 100;
			function = func;
		};
		double integrate(double begin, double end) {
			double dx = (end - begin) / h;

			double sum_odds = 0.0;
			for (int i = 1; i < h; i += 2)
			{
				sum_odds += function.function(begin + i * dx);
			}
			double sum_evens = 0.0;
			for (int i = 2; i < h; i += 2)
			{
				sum_evens += function.function(begin + i * dx);
			}
			return (function.function(begin) + function.function(end) + 2 * sum_evens + 4 * sum_odds) * dx / 3;
		}
	};
	class Fractal {
	public:
		double function(double x) {
			double fract = 1;
			for (int i = 1; i < x; i++)
			{
				fract *= x;
			}
			return fract;
		}
	};
	class Gamma {
	public:
		double function( double x ) {
			Fractal fractal;
			return fractal.function(x);
		}
	};
	
}
#endif