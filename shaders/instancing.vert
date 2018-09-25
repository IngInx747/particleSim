#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec4 instSpeedColor;
layout (location = 4) in mat4 instMatrix; // instance buffer

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;
out vec4 SpeedColor;

uniform mat4 uModel; // useless in this case
uniform mat4 uView;
uniform mat4 uProjection;

void main() {

	gl_Position = uProjection * uView * instMatrix * vec4(aPos, 1.0f);

	// Get one fragment's position in World Space
	FragPos = vec3(instMatrix * vec4(aPos, 1.0));

	// Also don't forget to transform normal vector
	Normal = mat3(transpose(inverse(instMatrix))) * aNormal;
	//Normal = mat3(instMatrix) * aNormal;

	TexCoords = aTexCoords;

	SpeedColor = instSpeedColor;
}