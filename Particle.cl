/** Particle.cl */

typedef struct __Particle_t
{
	float3 position;
	float3 velocity;
	float3 predicted_pos;
} Particle_t;

typedef struct __Lookup_t
{
	int offset;
	int size;
} Lookup_t;

////////////////////////////////////////////////////////////////////////////////////////////////////

// constant math
__constant float kInf = 1e20;
__constant float kEpsilon = 1e-3;
__constant float kPi = 3.14159265;

// constant control
__constant float delta_time = 0.01f;

// constant physics
__constant float gravity_accer = 9.8f;
__constant float density_water = 1.0f;

// constant particle
__constant float mass = 1.0f;
__constant float cutoff = 0.21f;

// bound faces indices
#define BB_RIGHT  0
#define BB_LEFT   1
#define BB_TOP    2
#define BB_BUTTOM 3
#define BB_FRONT  4
#define BB_BACK   5

// bound box sizes
#define BB_SCALE 1.8f
__constant float bb_sizes[6] = {
	 BB_SCALE, // right
	-BB_SCALE, // left
	 20.0f, // top
	-1.0f, // buttom
	 BB_SCALE, // front
	-BB_SCALE, // back
};

// bound box inner normal (normal toward INSIDE of bound box)
__constant float3 bb_normals[6] = {
	((float3) {-1.0f,  0.0f,  0.0f}), // right
	((float3) { 1.0f,  0.0f,  0.0f}), // left
	((float3) { 0.0f, -1.0f,  0.0f}), // top
	((float3) { 0.0f,  1.0f,  0.0f}), // buttom
	((float3) { 0.0f,  0.0f, -1.0f}), // front
	((float3) { 0.0f,  0.0f,  1.0f}), // back
};

// grid division
#define GRID_X 0
#define GRID_Y 1
#define GRID_Z 2
__constant int grid_dim[3] = {10, 10, 10};
__constant int grid_size = 1000;

__constant int3 grid_neighbors[27] = {
	((int3) {-1, -1, -1}), ((int3) { 0, -1, -1}), ((int3) { 1, -1, -1}),
	((int3) {-1,  0, -1}), ((int3) { 0,  0, -1}), ((int3) { 1,  0, -1}),
	((int3) {-1,  1, -1}), ((int3) { 0,  1, -1}), ((int3) { 1,  1, -1}),
	((int3) {-1, -1,  0}), ((int3) { 0, -1,  0}), ((int3) { 1, -1,  0}),
	((int3) {-1,  0,  0}), ((int3) { 0,  0,  0}), ((int3) { 1,  0,  0}),
	((int3) {-1,  1,  0}), ((int3) { 0,  1,  0}), ((int3) { 1,  1,  0}),
	((int3) {-1, -1,  1}), ((int3) { 0, -1,  1}), ((int3) { 1, -1,  1}),
	((int3) {-1,  0,  1}), ((int3) { 0,  0,  1}), ((int3) { 1,  0,  1}),
	((int3) {-1,  1,  1}), ((int3) { 0,  1,  1}), ((int3) { 1,  1,  1}),
};

#define MAX_NEIGHBORS 1000

////////////////////////////////////////////////////////////////////////////////////////////////////

float w_poly6(float radius, float div);
float w_spiky(float radius, float div);

float3 w_grad_poly6(float3 position, float div);
float3 w_grad_spiky(float3 position, float div);

int cell_3to1(int3 cell);
int3 cell_1to3(int cell_id);

int celling(float3 position);
bool out_of_grid(int3 cell);

int get_neighboring_particles(
	int* particle_table,
	float3 position,
	__constant Lookup_t* cell_lookup,
	__constant int* cell_ptc_table);

bool neighboring(float3 pos_1, float3 pos_2);

float3 reflect(float3 incidence, float3 normal);

int hitting_face(float3 vec);

bool bounding(Particle_t* particle);

__kernel void kernel_externel_force(__global Particle_t* particles);

__kernel void kernel_find_cell(__constant Particle_t* particles, __global int* cells);

__kernel void kernel_calc_lambda(
	__constant Particle_t* particles,
	__constant Lookup_t* cell_lookup,
	__constant int* cell_ptc_table,
	__global float* lambdas);

__kernel void kernel_calc_disp(
	__global Particle_t* particles,
	__constant Lookup_t* cell_lookup,
	__constant int* cell_ptc_table,
	__constant float* lambdas);

__kernel void kernel_update(__global Particle_t* particles);

__kernel void kernel_viscosity(
	__global Particle_t* particles,
	__constant Lookup_t* cell_lookup,
	__constant int* cell_ptc_table);

////////////////////////////////////////////////////////////////////////////////////////////////////

////////// externel forces //////////

__kernel void kernel_externel_force(__global Particle_t* particles)
{
	unsigned int index = get_global_id(0);

	// perform external force on particle
	particles[index].velocity.y += -gravity_accer * delta_time * mass;

	// predict position only affected by external forces
	particles[index].predicted_pos = particles[index].position + particles[index].velocity * delta_time;
}

////////// find neighbors //////////

__kernel void kernel_find_cell(__constant Particle_t* particles, __global int* cell_ids)
{
	unsigned int index = get_global_id(0);

	cell_ids[index] = celling(particles[index].predicted_pos);
}

////////// internel forces //////////

__kernel void kernel_calc_lambda(
	__constant Particle_t* particles,
	__constant Lookup_t* cell_lookup,
	__constant int* cell_ptc_table,
	__global float* lambdas)
{
	unsigned int index = get_global_id(0);
	unsigned int total = get_global_size(0);

	Particle_t particle = particles[index];

	float numerator = 0.0f;
	float denominator = 1.0f * kEpsilon;
	float ct = -0.00243f * kPi * density_water * pow(cutoff, 5);
	float3 self_grad = ((float3) {0.0f, 0.0f, 0.0f});

	int neighboring_particles[MAX_NEIGHBORS];
	int num_ptc = get_neighboring_particles(&neighboring_particles[0], particle.position, cell_lookup, cell_ptc_table);

	for (int i = 0; i < num_ptc; i++)
	{
		int ptc_id = neighboring_particles[i];
		float3 position = particle.predicted_pos - particles[ptc_id].predicted_pos;
		float radius = length(position);
		if (radius > cutoff) continue;
		float ratio = radius / cutoff;
		numerator += mass * pow(1.0f - ratio * ratio, 3);
		float inter_grad_scale = pow(1.0f - ratio, 4);
		denominator += inter_grad_scale;
		self_grad += inter_grad_scale * normalize(position);
	}

	denominator += dot(self_grad, self_grad);

	lambdas[index] = ct * (numerator / density_water - 1.0f) / denominator;
}

__kernel void kernel_calc_disp(
	__global Particle_t* particles,
	__constant Lookup_t* cell_lookup,
	__constant int* cell_ptc_table,
	__constant float* lambdas)
{
	unsigned int index = get_global_id(0);
	unsigned int total = get_global_size(0);

	Particle_t particle = particles[index];
	float lambda = lambdas[index];

	float3 displacement = ((float3) {0.0f,  0.0f,  0.0f});

	int neighboring_particles[MAX_NEIGHBORS];
	int num_ptc = get_neighboring_particles(&neighboring_particles[0], particle.position, cell_lookup, cell_ptc_table);

	for (int i = 0; i < num_ptc; i++)
	{
		int ptc_id = neighboring_particles[i];
		float3 position = particle.predicted_pos - particles[ptc_id].predicted_pos;
		float s_corr = 0.0f;
		s_corr = w_spiky(length(position), cutoff) / w_spiky(0, cutoff);
		s_corr = -0.01f * pow(s_corr, 4);
		displacement += w_grad_spiky(position, cutoff) * (lambda + lambdas[ptc_id] + s_corr);
	}

	particle.predicted_pos += displacement;

	bounding(&particle);

	particles[index].predicted_pos = particle.predicted_pos / density_water;
}

////////// update status of particles //////////

__kernel void kernel_update(__global Particle_t* particles)
{
	unsigned int index = get_global_id(0);

	Particle_t particle = particles[index];

	bool hitting_bound = bounding(&particle);

	particle.velocity = (particle.predicted_pos - particle.position) * (1.0f / delta_time);
	particle.position = particle.predicted_pos;

	particles[index].position = particle.position;
	particles[index].velocity = particle.velocity;
}

////////// fluid confinement //////////

__kernel void kernel_viscosity(
	__global Particle_t* particles,
	__constant Lookup_t* cell_lookup,
	__constant int* cell_ptc_table)
{
	unsigned int index = get_global_id(0);
	unsigned int total = get_global_size(0);

	Particle_t particle = particles[index];
	float3 viscosity = ((float3) {0.0f, 0.0f, 0.0f});

	// confining vorticity: viscosity
	//for (unsigned int i = 0; i < total; i++)
	//{
	//	if (neighboring(particle.predicted_pos, particles[i].predicted_pos, grid_div))
	//	{
	//		float3 position = particle.predicted_pos - particles[i].predicted_pos;
	//		float3 velocity = particle.velocity - particles[i].velocity;
	//		viscosity += 0.01f * w_spiky(length(position), cutoff) * velocity;
	//	}
	//}
	
	int neighboring_particles[MAX_NEIGHBORS];
	int num_ptc = get_neighboring_particles(&neighboring_particles[0], particle.position, cell_lookup, cell_ptc_table);

	for (int i = 0; i < num_ptc; i++)
	{
		int ptc_id = neighboring_particles[i];
		float3 position = particle.predicted_pos - particles[ptc_id].predicted_pos;
		float3 velocity = particle.velocity - particles[ptc_id].velocity;
		viscosity += 0.01f * w_spiky(length(position), cutoff) * velocity;
	}

	particles[index].velocity += viscosity;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int celling(float3 position)
{
	float div_x = (bb_sizes[BB_RIGHT] - bb_sizes[BB_LEFT])/ (float) grid_dim[GRID_X];
	float div_y = (bb_sizes[BB_TOP] - bb_sizes[BB_BUTTOM])/ (float) grid_dim[GRID_Y];
	float div_z = (bb_sizes[BB_FRONT] - bb_sizes[BB_BACK])/ (float) grid_dim[GRID_Z];

	int cell_x = floor((position.x - bb_sizes[BB_LEFT]) / div_x);
	int cell_y = floor((position.y - bb_sizes[BB_BUTTOM]) / div_y);
	int cell_z = floor((position.z - bb_sizes[BB_BACK]) / div_z);

	cell_x = clamp(cell_x, 0, grid_dim[GRID_X] - 1);
	cell_y = clamp(cell_y, 0, grid_dim[GRID_Y] - 1);
	cell_z = clamp(cell_z, 0, grid_dim[GRID_Z] - 1);

	return cell_x + cell_y * grid_dim[GRID_X] + cell_z * grid_dim[GRID_X] * grid_dim[GRID_Y];
}

bool out_of_grid(int3 cell)
{
	return cell.x < 0 || cell.x >= grid_dim[GRID_X] ||
		cell.y < 0 || cell.y >= grid_dim[GRID_Y] ||
		cell.z < 0 || cell.z >= grid_dim[GRID_Z];
}

int get_neighboring_particles(
	int* particle_table,
	float3 position,
	__constant Lookup_t* cell_lookup,
	__constant int* cell_ptc_table)
{
	int cell_id = celling(position);
	int3 cell = cell_1to3(cell_id);

	int count = 0;

	for (int i = 0; i < 27; i++)
	{
		int3 neighbor_cell = cell + grid_neighbors[i];
		if (out_of_grid(neighbor_cell)) continue;

		int neighbor_cell_id = cell_3to1(neighbor_cell);

		int offset = cell_lookup[neighbor_cell_id].offset;
		int num_ptc = cell_lookup[neighbor_cell_id].size;

		for (int j = 0; j < num_ptc; j++)
		{
			int ptc_id = cell_ptc_table[offset + j];
			particle_table[count] = ptc_id;
			count++;
			if (count >= MAX_NEIGHBORS) { return count; }
		}
	}

	return count;
}

int3 cell_1to3(int cell_id)
{
	return ((int3) {
		cell_id % grid_dim[GRID_X],
		(cell_id / grid_dim[GRID_X]) % grid_dim[GRID_Y],
		(cell_id / grid_dim[GRID_X] / grid_dim[GRID_Y]) % grid_dim[GRID_Z]
	});
}

int cell_3to1(int3 cell)
{
	return cell.x + cell.y * grid_dim[GRID_X] + cell.z * grid_dim[GRID_X] * grid_dim[GRID_Y];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// detect if two particles is neighboring
bool neighboring(float3 pos_1, float3 pos_2)
{
	float div_x = (bb_sizes[BB_RIGHT] - bb_sizes[BB_LEFT]) / (float) grid_dim[GRID_X];
	float div_y = (bb_sizes[BB_TOP] - bb_sizes[BB_BUTTOM]) / (float) grid_dim[GRID_Y];
	float div_z = (bb_sizes[BB_FRONT] - bb_sizes[BB_BACK]) / (float) grid_dim[GRID_Z];

	int grid_X_1 = floor(pos_1.x / div_x);
	int grid_Y_1 = floor(pos_1.y / div_y);
	int grid_Z_1 = floor(pos_1.z / div_z);

	int grid_X_2 = floor(pos_2.x / div_x);
	int grid_Y_2 = floor(pos_2.y / div_y);
	int grid_Z_2 = floor(pos_2.z / div_z);

	if (abs(grid_X_1 - grid_X_2) <= 1)
		if (abs(grid_Y_1 - grid_Y_2) <= 1)
			if (abs(grid_Z_1 - grid_Z_2) <= 1)
				return true;

	return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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

	cosine_axises[BB_RIGHT]  = dot(vec, -bb_normals[BB_RIGHT]);
	cosine_axises[BB_TOP]    = dot(vec, -bb_normals[BB_TOP]);
	cosine_axises[BB_FRONT]  = dot(vec, -bb_normals[BB_FRONT]);
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
	//float3 position = particle->position;
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
		//particle->velocity = mask * reflect(velocity, bb_normals[face]);
		particle->predicted_pos.x = clamp(predicted_pos.x, bb_sizes[BB_LEFT], bb_sizes[BB_RIGHT]);
		particle->predicted_pos.y = clamp(predicted_pos.y, bb_sizes[BB_BUTTOM], bb_sizes[BB_TOP]);
		particle->predicted_pos.z = clamp(predicted_pos.z, bb_sizes[BB_BACK], bb_sizes[BB_FRONT]);
		return true;
	} return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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
