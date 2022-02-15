#version 430


// Input from vertex shader
in vec2 texcoord;

// Output to the buffer
out vec4 p3d_FragColor;

// Declare uniforms we will use
uniform sampler2D p3d_Texture0;
uniform sampler2D sphereXYZ;
uniform uint level;
uniform uint c_maxSteps;

// Declare constants we will need
const float PI = 3.141592653589793;

//============================================================
float gCircleDistMag (in vec3 pos1, in vec3 pos2, out float dist)
{
    return -dot(pos1, pos2)/(dot(pos1, pos1) * dot(pos2, pos2));
}

//============================================================
// Credit to https://www.shadertoy.com/view/Mdy3DK
void main()
{
    float lev = clamp(level-1.0, 0.0, c_maxSteps);
    float stepwidth = floor(exp2(c_maxSteps - lev)+0.5);
    
    float bestDistance = 9999.0;
    vec2 bestSeedCoord = vec2(0.0);
    
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec4 sampleCoord = vec4(texcoord + vec2(x,y) * stepwidth, 0, 0);
            vec4 seedCoord = vec4(texture2D(p3d_Texture0, sampleCoord).rg, 0, 0);
            seedXYZ = texture2D(sphereXYZ, seedCoord).rgb * 2 - 255;
            callingXYZ = texture2D(sphereXYZ, texcoord.rg).rgb * 2 - 255;
            float dist = gCircleDistMag(seedXYZ, callingXYZ);
            if ((seedCoord.x != 0.0 || seedCoord.y != 0.0) && dist < bestDistance)
            {
                bestDistance = dist;
                bestSeedCoord = seedCoord;
            }
        }
    }
    p3d_FragColor = vec4(0.0);
    p3d_FragColor.rg = bestSeedCoord.rg
    p3d_FragColor.b = texture2D(output, p3d_Texture0).b
}