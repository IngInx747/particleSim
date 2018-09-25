#version 330 core

/** Directional Light */

struct Directional_Light_t {
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

vec3 CalcDirectionalLight(Directional_Light_t light, vec3 normal, vec3 viewDir,
	sampler2D diffuse, sampler2D specular, sampler2D emission);

/** Point Light */

struct  Point_Light_t {
	vec3 position;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float constant;
	float linear;
	float quadratic;
};

vec3 CalcPointLight(Point_Light_t light, vec3 normal, vec3 viewDir,
	sampler2D diffuse, sampler2D specular);

/** Spot Light */

struct Spot_Light_t {
	vec3 position;
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float constant;
	float linear;
	float quadratic;
	float innerCutOff;
	float outerCutOff;
};

vec3 CalcSpotLight(Spot_Light_t light, vec3 normal, vec3 viewDir,
	sampler2D diffuse, sampler2D specular);

/** Texture mapping */

struct MatTexMap_t {
	// texture diffuse
	sampler2D texture_diffuse1;
	sampler2D texture_diffuse2;
	sampler2D texture_diffuse3;
	sampler2D texture_diffuse4;
	// texture specular
	sampler2D texture_specular1;
	sampler2D texture_specular2;
	sampler2D texture_specular3;
	sampler2D texture_specular4;
	// texture normal
	sampler2D texture_normal1;
	sampler2D texture_normal2;
	// texture height
	sampler2D texture_height1;
	sampler2D texture_height2;
	// texture emission
	sampler2D texture_emission1;
	sampler2D texture_emission2;
	// To be added ...
};

/** Uniform variables */

// Camera
uniform vec3 uCameraPos;

// Lighting
#define NR_POINT_LIGHTS 4
uniform Directional_Light_t uDirectionalLight;
uniform Spot_Light_t uSpotLight;
uniform Point_Light_t uPointLights[NR_POINT_LIGHTS];

// Texture (Model Importer specified)
uniform MatTexMap_t uMaterial;

/** Stream variables */

out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
in vec4 SpeedColor; // Speed discriminator

void main() {

	vec3 normal = normalize(Normal);
	vec3 viewDir = normalize(uCameraPos - FragPos);
	vec3 resultColor = vec3(0.0, 0.0, 0.0);

	// Directional lighting
	resultColor += CalcDirectionalLight(uDirectionalLight, normal, viewDir,
		uMaterial.texture_diffuse1, uMaterial.texture_specular1, uMaterial.texture_emission1);

	// Spot lighting
	resultColor += CalcSpotLight(uSpotLight, normal, viewDir,
		uMaterial.texture_diffuse1, uMaterial.texture_specular1);

	// Point lighting
	/**
	for (int i=0; i<NR_POINT_LIGHTS; i++) {
		resultColor += CalcPointLight(uPointLights[i], normal, viewDir,
			uMaterial.texture_diffuse1, uMaterial.texture_specular1);
	}*/

	// Result
	FragColor = vec4(resultColor, 1.0) * SpeedColor;
}

vec3 CalcDirectionalLight(Directional_Light_t light, vec3 normal, vec3 viewDir,
	sampler2D diffuse, sampler2D specular, sampler2D emission) {

	vec3 lightDir = normalize(-light.direction);
	// ambient
	vec3 ambientColor = light.ambient * vec3(texture(diffuse, TexCoords));
	// diffuse
	float diffEff = max(dot(normal, lightDir), 0.0);
	vec3 diffuseColor = diffEff * light.diffuse * vec3(texture(diffuse, TexCoords));
	// specular
	vec3 reflectDir = reflect(-lightDir, normal);
	float specEff = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
	vec3 specularColor = specEff * light.specular * vec3(texture(specular, TexCoords));
	// emission
	vec3 emissionColor = vec3(0.0);
	if (texture(specular, TexCoords).r == 0.0)
		emissionColor = texture(emission, TexCoords).rgb;
	// result
	return ambientColor + diffuseColor + specularColor + emissionColor;
}

vec3 CalcPointLight(Point_Light_t light, vec3 normal, vec3 viewDir,
	sampler2D diffuse, sampler2D specular) {

	vec3 lightDir = normalize(light.position - FragPos);
	// Physics
	float distance = length(light.position - FragPos);
	float attenuation = 1.0 / (light.constant + light.linear*distance + light.quadratic*distance*distance);
	// ambient
	vec3 ambientColor = light.ambient * vec3(texture(diffuse, TexCoords));
	// diffuse
	float diffEff = max(dot(normal, lightDir), 0.0);
	vec3 diffuseColor = diffEff * light.diffuse * vec3(texture(diffuse, TexCoords));
	// specular
	vec3 reflectDir = reflect(-lightDir, normal);
	float specEff = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
	vec3 specularColor = specEff * light.specular * vec3(texture(specular, TexCoords));
	// result
	return attenuation * (ambientColor + diffuseColor + specularColor);
}

vec3 CalcSpotLight(Spot_Light_t light, vec3 normal, vec3 viewDir,
	sampler2D diffuse, sampler2D specular) {

	vec3 lightDir = normalize(light.position - FragPos);
	// Physics
	float distance = length(light.position - FragPos);
	float attenuation = 1.0 / (light.constant + light.linear*distance + light.quadratic*distance*distance);
	float theta = dot(lightDir, normalize(-light.direction));
	float epsilon = light.innerCutOff - light.outerCutOff;
	float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
	// Ambient lighting
	vec3 ambientColor = light.ambient * vec3(texture(diffuse, TexCoords));
	// Diffuse lighting
	float diffEff = max(dot(normal, lightDir), 0.0);
	vec3 diffuseColor = diffEff * light.diffuse * vec3(texture(diffuse, TexCoords));
	// Specular lighting
	vec3 reflectDir = reflect(-lightDir, normal);
	float specEff = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
	vec3 specularColor = specEff * light.specular * vec3(texture(specular, TexCoords));
	// Result lighting
	return attenuation * (ambientColor + (diffuseColor + specularColor) * intensity);
}
