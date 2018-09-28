/** Particle.cl */

typedef struct __Particle_t
{
	float3 position;
	float3 velocity;
	float3 predicted_pos;
} Particle_t;

//////////////////////////////////////////////////

// constant math
__constant float kInf = 1e20;
__constant float kEpsilon = 1e-3;
__constant float kPi = 3.14159265;

// constant control
__constant float delta_time = 0.01f;
__constant float grid_div = 0.3f; // [0.2 ~ 0.4]

// constant physics
__constant float gravity_accer = 9.8f;

//////////////////////////////////////////////////

float w_poly6(float radius, float div);
float w_spiky(float radius, float div);

float3 w_grad_poly6(float3 position, float div);
float3 w_grad_spiky(float3 position, float div);

bool neighboring(float3 pos_1, float3 pos_2, float div);

__kernel void kernel_calc_lambda(__constant Particle_t* particles, __global float* lambdas);

__kernel void kernel_calc_disp(__global Particle_t* particles, __constant float* lambdas);

//////////////////////////////////////////////////

__kernel void kernel_calc_disp(__global Particle_t* particles, __constant float* lambdas)
{
	unsigned int index = get_global_id(0);
	unsigned int total = get_global_size(0);

	Particle_t particle = particles[index];
	float lambda = lambdas[index];

	float3 displacement = ((float3) {0.0f,  0.0f,  0.0f});

	for (unsigned int i = 0; i < total; i++)
	{
		if (neighboring(particle.predicted_pos, particles[i].predicted_pos, grid_div))
		{
			float3 position = particle.predicted_pos - particles[i].predicted_pos;
			displacement += w_grad_spiky(position, grid_div) * (lambda + lambdas[i]);
		}
	}

	particles[index].predicted_pos += displacement * 0.00001f;
}

__kernel void kernel_calc_lambda(__constant Particle_t* particles, __global float* lambdas)
{
	unsigned int index = get_global_id(0);
	unsigned int total = get_global_size(0);

	Particle_t particle = particles[index];

	float numerator = 0.0f;
	float denominator = kEpsilon;
	float ct = -0.0364583f / grid_div;

	for (unsigned int i = 0; i < total; i++)
	{
		if (neighboring(particle.predicted_pos, particles[i].predicted_pos, grid_div))
		{
			float radius = length(particle.predicted_pos - particles[i].predicted_pos);
			if (radius > grid_div) continue;
			float ratio = radius / grid_div;
			numerator += pow(1.0f - ratio * ratio, 3);
			denominator += pow(1.0f - ratio, 4);
		}
	}

	lambdas[index] = ct * numerator / denominator;
}

// detect if two particles is neighboring
bool neighboring(float3 pos_1, float3 pos_2, float div)
{
	int grid_X_1 = floor(pos_1.x / div);
	int grid_Y_1 = floor(pos_1.y / div);
	int grid_Z_1 = floor(pos_1.z / div);

	int grid_X_2 = floor(pos_2.x / div);
	int grid_Y_2 = floor(pos_2.y / div);
	int grid_Z_2 = floor(pos_2.z / div);

	if (abs(grid_X_1 - grid_X_2) <= 1)
		if (abs(grid_Y_1 - grid_Y_2) <= 1)
			if (abs(grid_Z_1 - grid_Z_2) <= 1)
				return true;

	return false;
}

float w_poly6(float radius, float div)
{
	if (radius > div) return 0.0f;
	float ct = 315.0f / (64.0f * kPi);
	float div3 = pow(div, 3);
	float core = radius / div;
	core = 1.0f - core * core;
	core = core * core * core;
	return ct * core / div3;
}

float w_spiky(float radius, float div)
{
	if (radius > div) return 0.0f;
	float ct = 15.0f / kPi;
	float div3 = pow(div, 3);
	float core = radius / div;
	core = 1.0f - core;
	core = core * core * core;
	return ct * core / div3;
}

float3 w_grad_poly6(float3 position, float div)
{
	float radius = length(position);
	if (radius > div) return 0.0f;
	float ct = 945.0f / (32.0f * kPi);
	float div5 = pow(div, 5);
	float core = radius / div;
	core = 1.0f - core * core;
	core = core * core;
	core = -ct * core / div5;
	return position * core;
}

float3 w_grad_spiky(float3 position, float div)
{
	float radius = length(position);
	if (radius > div) return 0.0f;
	float ct = 45.0f / kPi;
	float div4 = pow(div, 4);
	float core = radius / div;
	core = 1.0f - core;
	core = core * core;
	core = -ct * core / (div4 * (radius + kEpsilon));
	return position * core;
}
