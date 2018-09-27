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

// constant physics
__constant float gravity_accer = 9.8f;

//////////////////////////////////////////////////

__kernel void kernel_main(__global Particle_t* particles);

//////////////////////////////////////////////////

__kernel void kernel_main(__global Particle_t* particles)
{
	unsigned int index = get_global_id(0);

	float3 position = particles[index].position;
	float3 velocity = particles[index].velocity;

	// perform external force on particle
	particles[index].velocity.y += -gravity_accer * delta_time * 1.0f;

	//
	particles[index].predicted_pos = position + velocity * delta_time;
}