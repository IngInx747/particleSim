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

// bound box inner normal (normal toward INSIDE of bound box)
__constant float3 bound_normals[6] = {
	((float3) {-1.0f,  0.0f,  0.0f}), // right
	((float3) { 1.0f,  0.0f,  0.0f}), // left
	((float3) { 0.0f, -1.0f,  0.0f}), // top
	((float3) { 0.0f,  1.0f,  0.0f}), // buttom
	((float3) { 0.0f,  0.0f, -1.0f}), // front
	((float3) { 0.0f,  0.0f,  1.0f}), // back
};

//////////////////////////////////////////////////

float3 reflect(float3 incidence, float3 normal);

int hitting_face(float3 vec);

void bounding(Particle_t* particle);

__kernel void kernel_main(__global Particle_t* particles);

//////////////////////////////////////////////////

__kernel void kernel_main(__global Particle_t* particles)
{
	unsigned int index = get_global_id(0);

	Particle_t particle = particles[index];

	bounding(&particle);
	
	particle.velocity = (particle.predicted_pos - particle.position) * (1.0f / delta_time);
	particle.position = particle.predicted_pos;

	particles[index].position = particle.position;
	particles[index].velocity = particle.velocity;
}

// generate a reflect vector based on incident and normal vectors
float3 reflect(float3 incidence, float3 normal)
{
	return incidence + normal * -2.0f * dot(incidence, normal);
}

// determine which face particle hits, return -1 if misses
int hitting_face(float3 vec)
{
	int face = -1; // which face vector hits
	float cosine_axises[6];
	float cosine_max = -kInf;

	cosine_axises[0] = dot(vec, -bound_normals[0]);
	cosine_axises[2] = dot(vec, -bound_normals[2]);
	cosine_axises[4] = dot(vec, -bound_normals[4]);
	cosine_axises[1] = -cosine_axises[0];
	cosine_axises[3] = -cosine_axises[2];
	cosine_axises[5] = -cosine_axises[4];
	
	for (int i = 0; i < 6; i++)
	{
		if (cosine_max < cosine_axises[i])
		{
			cosine_max = cosine_axises[i];
			face = i;
		}
	}

	if (cosine_max > 1.0f) return face;
	else return -1;
}

// perform bounding on fluid particles, reset whose params which exceed bound box
void bounding(Particle_t* particle)
{
	float3 position = particle->position;
	float3 velocity = particle->velocity;
	float3 predicted_pos = particle->predicted_pos;

	// detect bounding by predicted position
	int face = hitting_face(predicted_pos);

	if (face != -1)
	{
		float eff_collide = 0.8f;
		float3 mask = ((float3) {1.0f, 1.0f, 1.0f});
		if (face == 0 || face == 1) mask.x = eff_collide;
		if (face == 2 || face == 3) mask.y = eff_collide;
		if (face == 4 || face == 5) mask.z = eff_collide;
		particle->velocity = mask * reflect(velocity, bound_normals[face]);
		particle->predicted_pos.x = clamp(predicted_pos.x, -1.0f, 1.0f);
		particle->predicted_pos.y = clamp(predicted_pos.y, -1.0f, 1.0f);
		particle->predicted_pos.z = clamp(predicted_pos.z, -1.0f, 1.0f);
	}
}
