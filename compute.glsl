#version 430

// Set the number of invocations in the work group.
// We're going to use a single work group because jump flooding needs the entire texture
layout (local_size_x = 1024, local_size_y = 512) in;

// Declare uniforms we will use
uniform image2D tex;
uniform readonly image2D sphereXYZ;
uniform uint level;

// Declare constants we will need (2**10 = 1024 which is max texture dimension)
const uint c_maxSteps = 10;
const float PI = 3.141592653589793;

//============================================================
float gCircleDistMag (in vec3 pos1, in vec3 pos2, out float dist)
{
    float mag1 = length(pos1);
    float mag2 = length(pos2);
    float costheta = dot(pos1, pos2)/(mag1*mag2);
    return -costheta;
}

//============================================================
// Credit to https://www.shadertoy.com/view/Mdy3DK
vec4 StepJFA (in ivec2 texelCoord, out ivec3 newVal)
{
    float lev = clamp(level-1.0, 0.0, c_maxSteps);
    float stepwidth = floor(exp2(c_maxSteps - lev)+0.5);
    
    float bestDistance = 9999.0;
    vec2 bestCoord = vec2(0.0);
    
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 sampleCoord = ivec2(texelCoord + vec2(x,y) * stepwidth);
            vec2 seedCoord = imageLoad(tex, sampleCoord).rg;
            seedXYZ = imageLoad(sphereXYZ, seedCoord).xyz * 2 - 255;
            texelXYZ = imageLoad(sphereXYZ, texelCoord).xyz * 2 - 255;
            float dist = gCircleDistMag(seedXYZ, texelXYZ);
            if ((seedCoord.x != 0.0 || seedCoord.y != 0.0) && dist < bestDistance)
            {
                bestDistance = dist;
                bestCoord = seedCoord;
            }
        }
    }
    int seed = imageLoad(texIn, texelCoord);
    return ivec3(seed, bestCoord);
}

//============================================================
void main (void)
{
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec3 newVal = StepJFA(texelCoord)
    imageStore(tex, newVal.gb, newVal)
}