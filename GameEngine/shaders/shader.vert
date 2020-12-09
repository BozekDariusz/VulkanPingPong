#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0,binding = 0) uniform UniformBufferObjectGlobal{

	mat4 view;
	mat4 proj;

} globalUbo;

layout(set = 1,binding = 0) uniform UniformBufferObjectModel{

	mat4 model;

} modelUbo;

layout(location = 0) in vec2 inPosition;

layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 fragColor;


void main() {
    gl_Position = globalUbo.proj * globalUbo.view  *modelUbo.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}