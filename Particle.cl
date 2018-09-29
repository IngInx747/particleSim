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

// bound faces indices
#define BB_RIGHT  0
#define BB_LEFT   1
#define BB_TOP    2
#define BB_BUTTOM 3
#define BB_FRONT  4
#define BB_BACK   5

// bound box sizes
__constant float bound_sizes[6] = {
	 1.0f, // right
	-1.0f, // left
	 1.0f, // top
	-1.0f, // buttom
	 1.0f, // front
	-1.0f, // back
};

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

bool bounding(Particle_t* particle);

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

	cosine_axises[BB_RIGHT]  = dot(vec, -bound_normals[BB_RIGHT]);
	cosine_axises[BB_TOP]    = dot(vec, -bound_normals[BB_TOP]);
	cosine_axises[BB_FRONT]  = dot(vec, -bound_normals[BB_FRONT]);
	cosine_axises[BB_LEFT]   = -cosine_axises[BB_RIGHT];
	cosine_axises[BB_BUTTOM] = -cosine_axises[BB_TOP];
	cosine_axises[BB_BACK]   = -cosine_axises[BB_FRONT];
	
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
bool bounding(Particle_t* particle)
{
	float3 position = particle->position;
	float3 velocity = particle->velocity;
	float3 predicted_pos = particle->predicted_pos;

	// detect bounding by predicted position
	int face = hitting_face(predicted_pos);

	if (face != -1)
	{
		//float eff_collide = 0.5f;
		//float3 mask = ((float3) {1.0f, 1.0f, 1.0f});
		//if (face == BB_RIGHT || face == BB_LEFT) mask.x = eff_collide;
		//if (face == BB_TOP || face == BB_BUTTOM) mask.y = eff_collide;
		//if (face == BB_FRONT || face == BB_BACK) mask.z = eff_collide;
		//particle->velocity = mask * reflect(velocity, bound_normals[face]);
		particle->predicted_pos.x = clamp(predicted_pos.x, bound_sizes[BB_LEFT], bound_sizes[BB_RIGHT]);
		particle->predicted_pos.y = clamp(predicted_pos.y, bound_sizes[BB_BUTTOM], bound_sizes[BB_TOP]);
		particle->predicted_pos.z = clamp(predicted_pos.z, bound_sizes[BB_BACK], bound_sizes[BB_FRONT]);
		return true;
	} return false;
}
