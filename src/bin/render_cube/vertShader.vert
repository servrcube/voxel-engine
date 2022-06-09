#version 450

                
layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform Data {
    mat4 view;
} uniforms;


void main() {
    gl_Position = uniforms.view * vec4(position.xy, 0.0, 1.0);
}