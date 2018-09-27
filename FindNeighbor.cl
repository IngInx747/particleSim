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

// constant control
__constant float delta_time = 0.01f;
__constant float grid_div = 0.3f; // [0.2 ~ 0.4]

// constant physics
__constant float gravity_accer = 9.8f;

//////////////////////////////////////////////////

bool neighboring(float3 pos_1, float3 pos_2);

__kernel void kernel_main(__global Particle_t* particles);

//////////////////////////////////////////////////

__kernel void kernel_main(__global Particle_t* particles)
{
	unsigned int index = get_global_id(0);
	unsigned int total = get_global_size(0);

	Particle_t particle = particles[index];

	for (unsigned int i = 0; i < total; i++)
	{
		if (neighboring(particle.position, particles[i].position))
		{
			float dist = length(particle.position - particles[i].position);
			dist = pow(dist, 6);
			particles[index].velocity += -dist * (particle.position - particles[i].position);
		}
	}
}

// detect if two particles is neighboring
bool neighboring(float3 pos_1, float3 pos_2)
{
	int grid_X_1 = floor(pos_1.x / grid_div);
	int grid_Y_1 = floor(pos_1.y / grid_div);
	int grid_Z_1 = floor(pos_1.z / grid_div);

	int grid_X_2 = floor(pos_2.x / grid_div);
	int grid_Y_2 = floor(pos_2.y / grid_div);
	int grid_Z_2 = floor(pos_2.z / grid_div);

	if (abs(grid_X_1 - grid_X_2) <= 1)
		if (abs(grid_Y_1 - grid_Y_2) <= 1)
			if (abs(grid_Z_1 - grid_Z_2) <= 1)
				return true;

	return false;
}
