#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <functional>
#include <chrono>
#include <omp.h>
#include <unordered_map>
#define _USE_MATH_DEFINES
 


//dt

float fixedDt = 1 / 120.0f;


const unsigned int screenWidth = 1800;
const unsigned int screenHeight = 900;

static double fuc_ms_avg = 0.0;
static int fuc_samples = 0;
static double fuc_ms = 0.0;

struct View {
    float cx = screenWidth * 0.5f;
    float cy = screenHeight * 0.5f;
    float height = (float)screenHeight;
    float aspect = (float)screenWidth / (float)screenHeight;
    float zoom = 1.0f;
    float width() const { return height * aspect; }
} view;
struct Vec3 {
    float x, y, z;
};
// dumb operators
//perf killers
inline Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
 Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Vec3 operator*(Vec3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
inline Vec3 operator/(Vec3 a, float s) { return { a.x / s, a.y / s, a.z / s }; }
inline Vec3 operator*(float s, Vec3 a) {
    return { a.x * s, a.y * s, a.z * s };
}

inline Vec3 operator-(Vec3 v) {
    return { -v.x, -v.y, -v.z };
}
inline Vec3& operator+=(Vec3& a, const Vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline Vec3& operator-=(Vec3& a, const Vec3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
inline Vec3& operator*=(Vec3& v, float s) {
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline float length(Vec3 v) {
    return sqrt(dot(v, v));
}

inline Vec3 normalize(Vec3 v) {
    float l = length(v);
    return (l > 0.0f) ? v / l : Vec3{ 0,0,0 };
}
inline Vec3 normalizeVec3(Vec3 v) {
    float l = length(v);
    return (l > 0.0f) ? v / l : Vec3{ 0,0,0 };
}

//slow-ass struct
struct Body {


    Vec3 pos;
    Vec3 acl;
    Vec3 vel;
   
    Vec3 acc_old;
    float size;
    float mass;
  
    bool iscenter;
    int r, b, g;
    bool alive;
    float heat = 0.0f;
    Vec3 force;
    float density;
    int br, bg, bb;
    float pressure;


};





//perf killer
std::vector<Body> bodies;
 
//useless
const float M_PI = 3.14159265359f;

constexpr float MAX_HEAT = 100.0f;
constexpr float HEAT_TO_COLOR = 2.0f;


struct Camera {
    glm::vec3 position = glm::vec3(0.0f, -200.0f, 21.0f); // cinematic 3D
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;

    float yaw = 90.0f;   // diagonal
    float pitch = -15.0f;  // looking down
    float fov = 70.0f;
};

Camera camera;
bool firstMouse = true;
double lastMouseX = 0, lastMouseY = 0;
bool cameraRotating = false;

float mouseSensitivity = 0.15f;
float scrollSensitivity = 2.0f;

bool mouseMassActive = false;
int mouseMassIndex = -1;   // index in bodies vector

// CHANGE: OpenGL vertex structure for batched rendering
struct GLVertex {
    float px, py,pz;           // world position
    float radius;           // radius in screen pixels
    float cr, cg, cb, ca;   // color
    float ox, oy;  
    
    // quad corner offset
};

// CHANGE: OpenGL resources
GLuint vao = 0, vbo = 0, ibo = 0;
size_t vbo_capacity = 0;
size_t ibo_capacity = 0;
GLuint program = 0;

const char* vertexShaderSource = R"glsl(
#version 330 core

layout(location = 0) in vec3 inCenterWorld;
layout(location = 1) in float inRadius;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inOffset;
layout(location = 4) in float inHeat;   // b.heat ∈ [0,100]

uniform mat4 uProj;
uniform mat4 uView;

out vec4 vColor;
out vec2 vOffset;
out float vHeat;

void main() {
    vec3 right = vec3(uView[0][0], uView[1][0], uView[2][0]);
    vec3 up    = vec3(uView[0][1], uView[1][1], uView[2][1]);

    vec3 worldPos =
        inCenterWorld +
        (right * inOffset.x + up * inOffset.y) * inRadius;

    gl_Position = uProj * uView * vec4(worldPos, 1.0);

    vColor = inColor;
    vOffset = inOffset; // ✅ FIXED
 vHeat   = inHeat;
}


)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core

in vec2 vOffset;
in vec4 vColor;
in float vHeat;

out vec4 FragColor;

// direction TOWARDS the light (normalized)
uniform vec3 uLightDir;

void main() {
    // ---- circle cutout ----
    float r2 = dot(vOffset, vOffset);
    if (r2 > 1.0) discard;

    // ---- fake sphere normal ----
    float z = sqrt(1.0 - r2);
    vec3 normal = normalize(vec3(vOffset, z));

    // ---- lighting ----
    float diff = max(dot(normal, uLightDir), 0.0);
    float light = 0.3 + 0.7 * diff;

    vec3 baseColor = vColor.rgb * light;

    // ---- HEAT-DRIVEN CHEAP BLOOM ----
    // normalize heat: 0–100 → 0–1
    float heat01 = clamp(vHeat / 100.0, 0.0, 1.0);

    // perceived brightness
    float brightness = dot(baseColor, vec3(0.2126, 0.7152, 0.0722));

    // hot stuff blooms easier
    float threshold = mix(0.75, 0.35, heat01);

    // soft glow mask
    float glow = smoothstep(threshold, 1.0, brightness);

    // bloom amount (tweak 2.0 if you want more punch)
    vec3 bloom = baseColor * glow * heat01 * 5.0;

    FragColor = vec4(baseColor + bloom, 1.0);
}


)glsl";

// CHANGE: Shader compilation helper
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetShaderInfoLog(s, 1024, nullptr, buf);
        std::cerr << "Shader compile error: " << buf << "\n";
    }
    return s;
}

static GLuint createProgram(const char* vs, const char* fs) {
    GLuint a = compileShader(GL_VERTEX_SHADER, vs);
    GLuint b = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, a);
    glAttachShader(p, b);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(p, 1024, nullptr, buf);
        std::cerr << "Program link error: " << buf << "\n";
    }
    glDeleteShader(a); glDeleteShader(b);
    return p;
}

void ensureVBOCapacity(size_t verts) {
    if (verts <= vbo_capacity) return;

    vbo_capacity = verts * 2 + 256;
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao == 0) glGenVertexArrays(1, &vao);

    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vbo_capacity * sizeof(GLVertex), nullptr, GL_STREAM_DRAW);

    GLsizei stride = sizeof(GLVertex);

    // Attribute 0: world position (px, py)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
        (void*)offsetof(GLVertex, px));
    // Attribute 1: radius (removed screen center, now just radius)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(GLVertex, radius));

    // Attribute 2: color (cr, cg, cb, ca)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(GLVertex, cr));

    // Attribute 3: offset (ox, oy)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(GLVertex, ox));


    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// CHANGE: IBO for indexed quad rendering
void ensureIBOCapacity(size_t numBodies) {
    size_t numIndices = numBodies * 6;
    if (numIndices <= ibo_capacity) return;

    ibo_capacity = numIndices * 2;
    if (ibo) glDeleteBuffers(1, &ibo);

    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, ibo_capacity * sizeof(GLuint), nullptr, GL_STATIC_DRAW);

    std::vector<GLuint> indices(ibo_capacity);
    for (size_t i = 0; i < ibo_capacity / 6; i++) {
        GLuint base = i * 4;
        indices[i * 6 + 0] = base + 0;
        indices[i * 6 + 1] = base + 1;
        indices[i * 6 + 2] = base + 2;
        indices[i * 6 + 3] = base + 0;
        indices[i * 6 + 4] = base + 2;
        indices[i * 6 + 5] = base + 3;
    }
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, indices.size() * sizeof(GLuint), indices.data());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


// CHANGE: Orthographic projection matrix
void setOrtho(float left, float right, float bottom, float top, float nearv, float farv, float* out4x4) {
    float dx = right - left;
    float dy = top - bottom;
    float dz = farv - nearv;
    for (int i = 0; i < 16; ++i) out4x4[i] = 0.0f;
    out4x4[0] = 2.0f / dx;
    out4x4[5] = 2.0f / dy;
    out4x4[10] = -2.0f / dz;
    out4x4[12] = -(right + left) / dx;
    out4x4[13] = -(top + bottom) / dy;
    out4x4[14] = -(farv + nearv) / dz;
    out4x4[15] = 1.0f;
}

// CHANGE: World to screen coordinate conversion
inline void worldToScreen_topLeft(float wx, float wy, float& sx, float& sy, const View& v) {
    float halfH = v.height * 0.5f * v.zoom;
    float halfW = v.width() * 0.5f * v.zoom;
    float left = v.cx - halfW;
    float top = v.cy - halfH;
    float scale = (float)screenHeight / (v.height * v.zoom);
    sx = (wx - left) * scale;
    sy = (wy - top) * scale;
}



inline float randf(float min, float max) {
    return min + (max - min) * (float(rand()) / float(RAND_MAX));
}


Vec3 randomInSphere(float r) {
    float u = (float)rand() / RAND_MAX;
    float v = (float)rand() / RAND_MAX;
    float w = (float)rand() / RAND_MAX;

    float theta = u * 2.0f * M_PI;
    float phi = acos(2.0f * v - 1.0f);
    float rad = cbrtf(w) * r; // uniform volume

    return {
        rad * sin(phi) * cos(theta),
        rad * sin(phi) * sin(theta),
        rad * cos(phi)
    };


}////////////////////////////////////////////////////////////////////////
//ui////////////
//functions
bool colisionFun = true;
bool gravityFun = true;
bool updateFun = true;

//simspeed
float simspeed = 1.0f;


//formation mode
int mode = 0;

//particle settings
float G = 6.674f;
int totalBodies = 1000;
float size = 1.0f;

//star settings
float centermass = 1000.0f;
float setMass = 1.0f;
float starsize = size * 3;

//orbit settings
float radius = 100.0f;
float Maxradius = 200.0f;
float orbitalspeed = 1.0f;

//omp
bool omp1 = true;
bool omp2 = true;
bool omp3 = true;
bool omp4 = true;
bool omp5 = false;
bool omp6 = false;
int substeps = 1;
int cc1 = 16;

//camera
float y, p;

//heat
float cold = 0.500f;
float cr = 2.0f;
float hmulti = 15.0f;

//fluid grid cords
int nx = 50;
int ny = 50;
int nz = 50;

//world cords
float wx, wy, wz;

//modes setting
float impactSpeed=2.50f;
float DST = 100.0f;
float yspeed = 2.0f;
//sph variables
float h = 2.50f; 
float h2 = h * h;
float rest_density = 0.60f;//density-idk
float pressure = 200.0f;//pressure-idk--K
float alpha_visc = 1.3f;  // Viscosity strength (0.5-2.0)
float beta_visc = 2.0f;   // Shock capturing
float gamma = 1.3f;
bool lockstar = false;
int maxX = 500;
int maxY = 500;
bool heateffect = true;



//////////////////////////////////////
//register particles-very unprofessional
void registerBody(int totalBodies, float size, float setMass, float radius, float orbitalspeed) {
    bodies.clear();
    bodies.reserve(totalBodies + 1);
   


    for (int i = 0; i < totalBodies; ++i) {
        Body& b = bodies[i];
        b.size = size;
        b.mass = 1.0f;
        b.iscenter = false;

        float x = 0, y = 0, z = 0;
        float vx = 0, vy = 0, vz = 0;

       
       
        if (mode == 0) {
            if (i == 0) {
                b.mass = centermass;
                b.size = starsize;
                b.iscenter = true;
            }
            else {
                float minR = radius;
                float maxR = Maxradius ;

                float angle = randf(0, 2 * M_PI);
                float r = randf(minR, maxR);

                x = cos(angle) * r;
                y = sin(angle) * r;
                z = randf(-2.5f, 2.5f);

                float v = sqrt(G * centermass / r) * orbitalspeed;
                vx = sin(angle) * v;
                vy = -cos(angle) * v;
                vz = randf(-0.05f, 0.05f) * v;
            }
        }

        // ===============================
        // MODE 1 — random chaos
        // ===============================
        if (mode == 1) {
            x = randf(0,maxX);
            y = randf(0,maxY);
            z = randf(-5, 5);
            int n = totalBodies / 10;
            if(i < 100 ) {
                b.mass = 2.5;
            }
           
            vx = randf(-5, 5);
            vy = randf(-5, 5);
            vz = 0;
        }

        // ===============================
        // MODE 2 — double star system
        // ===============================
        if (mode == 2) {
            int half = totalBodies / 2;

            Vec3 c1 = { screenWidth * 0.3f, screenHeight * 0.5f, 0 };
            Vec3 c2 = { screenWidth * 0.7f, screenHeight * 0.5f, 0 };

            if (i == 0 || i == half) {
                b.mass = centermass;
                b.size = starsize;
                b.iscenter = true;

                Vec3 c = (i == 0) ? c1 : c2;
                x = c.x; y = c.y; z = 0;
            }
            else {
                Vec3 c = (i < half) ? c1 : c2;

                float angle = randf(0, 2 * M_PI);
                float r = randf(radius, Maxradius);

                x = c.x + cos(angle) * r;
                y = c.y + sin(angle) * r;
                z = randf(-3, 3);

                float v = sqrt(G * centermass / r);
                vx = -sin(angle) * v;
                vy = cos(angle) * v;
            }
        }

        // ===============================
        // MODE 3 — STAR + SPHERICAL CLOUD (NEW 🔥)
        // ===============================
        if (mode == 3) {
            if (i == 0) {
                b.mass = centermass;
                b.size = starsize;
                b.iscenter = true;
                x = 0; y = 0; z = 0;
            }
            else {
                float u = (float)rand() / RAND_MAX;
                float v = (float)rand() / RAND_MAX;
                float w = (float)rand() / RAND_MAX;

                float theta = u * 2.0f * M_PI;
                float phi = acos(2.0f * v - 1.0f);
                float rad = cbrtf(w) * radius*5; // uniform volume

                Vec3 p = {
    rad * sinf(phi) * cosf(theta),
    rad * sinf(phi) * sinf(theta),
    rad * cosf(phi)
                };
                x = p.x;
                y = p.y;
                z = p.z;

                
                vx = randf(-0.1f, 0.1f);
                vy = randf(-0.1f, 0.1f);
                vz = randf(-0.1f, 0.1f);
            }
        }
        if (mode == 4)
        {
            int earthCount = (int)(totalBodies * 0.85f);
            int theiaCount = totalBodies - earthCount;
            float particle_spacing = h*0.8f;  // matches size

            if (i < earthCount)
            {
                // EARTH - cubic grid
                int particles_per_side = (int)ceil(cbrt((float)earthCount));
                int ix = i % particles_per_side;
                int iy = (i / particles_per_side) % particles_per_side;
                int iz = i / (particles_per_side * particles_per_side);

                x = ix * particle_spacing;
                y = iy * particle_spacing;
                z = iz * particle_spacing;

                vx = vy = vz = 0.0f;
            }
            else
            {
                // THEIA - cubic grid, positioned far away
                int theiaIndex = i - earthCount;
                int particles_per_side = (int)ceil(cbrt((float)theiaCount));
                int ix = theiaIndex % particles_per_side;
                int iy = (theiaIndex / particles_per_side) % particles_per_side;
                int iz = theiaIndex / (particles_per_side * particles_per_side);

                // Calculate Earth's width
                int earth_side = (int)ceil(cbrt((float)earthCount));
                float earth_width = earth_side * particle_spacing;

                // Place Theia far away (50 units separation)
                float separation = earth_width + DST;

                x = separation + ix * particle_spacing;
                y = iy * particle_spacing;
                z = iz * particle_spacing;

                vx = -impactSpeed * cos(45);
                vy = yspeed*sin(45);
                vz = 0.0f;
            }

            b.mass = 1.0f;
            b.size = size;
            b.heat = 0.0f;
        }
       
        b.density = rest_density;   // NOT 0
        b.pressure = 0.0f;
        b.pos = { x, y, z };
        b.vel = { vx, vy, vz };
        b.acl = { 0,0,0 };
        b.alive = true;
        b.br = 20;
        b.bg = 45;
        b.bb = 220;
        
        bodies.push_back(b);
    }

}
void restartSimulation(int totalbodies, float size, float setMass, float radius, float orbitalspeed) {
    bodies.clear();
    registerBody(totalbodies, size, setMass, radius, orbitalspeed);
   
}//
//i hate trees :xxx
struct BHNode {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    Vec3 c;
    Vec3 com;
    float mass;
    // center of mass
    BHNode* children[8];

    Body* body;   // single body (leaf)

    BHNode(float minx, float miny, float minz,
        float maxx, float maxy, float maxz)
        : minX(minx), minY(miny), minZ(minz),
        maxX(maxx), maxY(maxy), maxZ(maxz),
        mass(0), com(0,0,0),
        body(nullptr) {
        c = {
            0.5f * (minX + maxX),
            0.5f * (minY + maxY),
            0.5f * (minZ + maxZ)
        };

        for (int i = 0; i < 8; i++) children[i] = nullptr;
    }

    bool isLeaf() const {
        return children[0] == nullptr;
    }

    float size() const {
        return maxX - minX; // cube
    }
};
inline int octant(BHNode* n, Body* b) {
    int o = 0;
    if (b->pos.x > n->c.x) o |= 1;
    if (b->pos.y > n->c.y) o |= 2;
    if (b->pos.z > n->c.z) o |= 4;
    return o;
}
void subdivide(BHNode* n) {
    float mx = n->c.x;
    float my = n->c.y;
    float mz = n->c.z;

    for (int i = 0; i < 8; i++) {
        float x0 = (i & 1) ? mx : n->minX;
        float x1 = (i & 1) ? n->maxX : mx;
        float y0 = (i & 2) ? my : n->minY;
        float y1 = (i & 2) ? n->maxY : my;
        float z0 = (i & 4) ? mz : n->minZ;
        float z1 = (i & 4) ? n->maxZ : mz;

        n->children[i] = new BHNode(x0, y0, z0, x1, y1, z1);
    }
}
void insert(BHNode* n, Body* b) {
    if (n->isLeaf()) {
        if (!n->body) {
            n->body = b;
        }
        else {
            Body* old = n->body;
            n->body = nullptr;
            subdivide(n);
            insert(n->children[octant(n, old)], old);
            insert(n->children[octant(n, b)], b);
        }
    }
    else {
        insert(n->children[octant(n, b)], b);
    }
}
BHNode* buildTree() {
    float minX = bodies[0].pos.x, maxX = bodies[0].pos.x;
    float minY = bodies[0].pos.y, maxY = bodies[0].pos.y;
    float minZ = bodies[0].pos.z, maxZ = bodies[0].pos.z;

   for (int i = 0; i < bodies.size(); i++) {
        const Body& b = bodies[i];
        minX = std::min(minX, b.pos.x);
        maxX = std::max(maxX, b.pos.x);
        minY = std::min(minY, b.pos.y);
        maxY = std::max(maxY, b.pos.y);
        minZ = std::min(minZ, b.pos.z);
        maxZ = std::max(maxZ, b.pos.z);
    }


    float span = std::max({ maxX - minX, maxY - minY, maxZ - minZ }) + 1.0f;
    float cx = 0.5f * (minX + maxX);
    float cy = 0.5f * (minY + maxY);
    float cz = 0.5f * (minZ + maxZ);

    BHNode* root = new BHNode(
        cx - span, cy - span, cz - span,
        cx + span, cy + span, cz + span
    );

    for (auto& b : bodies)
        insert(root, &b);

    return root;
}
void computeMass(BHNode* n) {
    if (!n) return;

    if (n->isLeaf()) {
        if (n->body) {
            n->mass = n->body->mass;
            n->com = n->body->pos;
        }
        return;
    }

    float m = 0.0f;
    Vec3 csum = { 0,0,0 };
  for (int i = 0; i < 8; i++) {
        computeMass(n->children[i]);
        BHNode* c = n->children[i];
        m += c->mass;
        csum += c->com * c->mass;
        
    }

    if (m > 0) {
        n->mass = m;
        n->mass = m;
        n->com = csum * (1.0f / m);
    }
}
void deleteTree(BHNode* n) {
    if (!n) return; 
   for (int i = 0; i < 8; i++) {
        if (n->children[i])
            deleteTree(n->children[i]);
    }

    delete n;
}
 float theta = 0.3f;
 float eps2 = 0.5f*h;
float planetMassThreshold = 50.0f * 1;
void applyForce(Body& b, BHNode* n) {
    if (!n || n->mass == 0) return;
    if (n->isLeaf() && n->body == &b)
        return;
    Vec3 r = n->com - b.pos;
    float r2 = dot(r, r) + eps2;
    float d = sqrtf(r2);

    if (n->isLeaf() || (n->size() / d) < theta) {
        float invR3 = 1.0f / (r2 * d);
        Vec3 a = r * (G * n->mass * invR3);
        b.acl += a;
    }
    else {
      for (int i = 0; i < 8; i++)
            applyForce(b, n->children[i]);
    }
}
void gravity() {
    
        

    BHNode* root = buildTree();
    computeMass(root);

#pragma omp parallel for if(omp1) schedule(dynamic,16)
    for (int i = 0; i < bodies.size(); i++)
        applyForce(bodies[i], root);

    deleteTree(root);
   

}
struct ThreadSafeGrid {
    std::vector<std::unordered_map<int, std::vector<int>>> thread_grids;
    int num_threads;

    ThreadSafeGrid() {
        num_threads = omp_get_max_threads();
        thread_grids.resize(num_threads);
    }

    void clear() {
        for (auto& grid : thread_grids) {
            grid.clear();
        }
    }

    void insert(int thread_id, int hash, int body_idx) {
        thread_grids[thread_id][hash].push_back(body_idx);
    }

    // Merge all thread grids into one
    std::unordered_map<int, std::vector<int>> merge() {
        std::unordered_map<int, std::vector<int>> final_grid;

        for (const auto& tgrid : thread_grids) {
            for (const auto& [hash, indices] : tgrid) {
                auto& vec = final_grid[hash];
                vec.insert(vec.end(), indices.begin(), indices.end());
            }
        }

        return final_grid;
    }
};
ThreadSafeGrid ts_grid;
struct cellcoord { int x, y, z; };
inline int hashcell(int x, int y, int z) {
    const int p1 = 73856093;
    const int p2 = 19349663;
    const int p3 = 83492791;
    return (x * p1) ^ (y * p2) ^ (z * p3);
}
std::unordered_map<int, std::vector<int >> grid;
inline cellcoord getcell(const Vec3& p) {
    return{
        (int)floor(p.x / h),
        (int)floor(p.y / h),
        (int)floor(p.z / h)
    };
}
void buildgrid(std::vector<Body>& bodies) {
    ts_grid.clear();

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

#pragma omp for schedule(static)
        for (int i = 0; i < bodies.size(); i++) {
            cellcoord c = getcell(bodies[i].pos);
            int key = hashcell(c.x, c.y, c.z);
            ts_grid.insert(tid, key, i);
        }
    }

    // Merge thread-local grids
    grid = ts_grid.merge();
}
template<typename func>
void foreachneighbor(int i, const std::vector<Body>& bodies, func fun) {
    cellcoord c = getcell(bodies[i].pos);
    for(int dx =-1;dx<=1;dx++)
    for(int dy =-1;dy<=1;dy++)
        for (int dz = -1; dz <= 1; dz++) {
            int key = hashcell(c.x + dx, c.y + dy, c.z + dz);

            auto it = grid.find(key);
            if (it == grid.end())continue;

            for (int j : it->second) {
                if(j == i) continue;
                fun(j);
            }
    }
}
float smoothingkernel(float r, float h) {
    if (r >= 0.0f && r < h) {
        float poly6coeff = 315.0f / (64.0f * M_PI * pow(h, 9));

        float v = h * h - r * r;
        return poly6coeff * v * v * v;
    }
    return 0.0f;

  
}
float spikyKernel(float r, float h) {
    if (r >= 0.0f && r < h) {
        float coeff = 15.0f / (M_PI * pow(h, 6));
        float x = h - r;
        return coeff * x * x * x;
    }
    return 0.0f;
}
float densitykernel(float dst, float radius) {
   
    return smoothingkernel(dst, radius);
   // return spikyKernel(dst, radius);
}
float spikyGrad(float r, float h) {
    if (r > 0.0f && r < h) {
        float v = h - r;
        return -45.0f / (M_PI * pow(h, 6)) * v * v;
    }
    return 0.0f;
}
float PressureFromDensity(float density)
{
   // return std::max(0.0f, pressure * (density - rest_density));

    // Adiabatic index
    return pressure * powf(density/rest_density , gamma);

}
float artificialViscosity(const Body& i, const Body& j, float r, float h) {
    Vec3 vij = i.vel - j.vel;
    Vec3 rij = i.pos - j.pos;

    float vdotr = dot(vij, rij);
    if (vdotr >= 0.0f) return 0.0f; // particles moving apart

    float rho_avg = 0.5f * (i.density + j.density);
    float c_s = sqrtf(gamma * pressure * powf(rho_avg / rest_density, gamma - 1.0f));

    float mu = h * vdotr / (r * r + 0.01f * h * h);

    return (-alpha_visc * c_s * mu + beta_visc * mu * mu) / rho_avg;
}
float viscosityKernel(float r, float h) {
    if (r < h) {
        return 45.0f / (M_PI * pow(h, 6)) * (h - r);
    }
    return 0.0f;
}
void computedensity() {
    const float h2_local = h * h;

    
#pragma omp parallel for if(omp2)schedule(dynamic,64)
    for (int i = 0; i < bodies.size(); i++) {
        float density = 0.0f;
        const Vec3& pos_i = bodies[i].pos;

        cellcoord c = getcell(pos_i);
        density += bodies[i].mass * densitykernel(0.0f, h);

        // Search neighboring cells
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int key = hashcell(c.x + dx, c.y + dy, c.z + dz);

                    auto it = grid.find(key);
                    if (it == grid.end()) continue;

                    for (int j : it->second) {
                        if (j == i) continue;

                        Vec3 r = pos_i - bodies[j].pos;
                        float r2 = dot(r, r);

                        if (r2 < h2_local) {
                            float dst = sqrtf(r2);
                            density += bodies[j].mass * densitykernel(dst, h);
                        }
                    }
                }
            }
        }
       
       

      

        
        bodies[i].density = std::max(density, 1e-6f);
    }
}
void coputepressureforce(float dt) {
    const float h2_local = h * h;

#pragma omp parallel for if(omp3)schedule(dynamic,64)
    for (int i = 0; i < bodies.size(); i++) {
        Vec3 force = { 0,0,0 };
        const Vec3& pos_i = bodies[i].pos;
        float rho_i = bodies[i].density;
        float p_i = PressureFromDensity(rho_i);
        Vec3 artificialVisc = { 0,0,0 };


        cellcoord c = getcell(pos_i);

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int key = hashcell(c.x + dx, c.y + dy, c.z + dz);

                    auto it = grid.find(key);
                    if (it == grid.end()) continue;

                    for (int j : it->second) {
                        if (j == i) continue;

                        Vec3 r = pos_i - bodies[j].pos;
                        float r2 = dot(r, r);

                        if (r2 < h2_local && r2 > 1e-12f) {
                            float rLen = sqrtf(r2);
                            float rho_j = bodies[j].density;

                            if (rho_j < 1e-6f) continue;

                            float p_j = PressureFromDensity(rho_j);
                            Vec3 dir = r / rLen;
                            float gradW = spikyGrad(rLen, h);

                            // Symmetric pressure force
                            float pressureTerm = (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j));
                            force -= bodies[j].mass * pressureTerm * gradW * dir;
                            float pi_ij = artificialViscosity(bodies[i], bodies[j], rLen, h);
                            artificialVisc -= bodies[j].mass * pi_ij * gradW * dir;

                        }

                    

                    }
                }
            }
        }

        bodies[i].force = (force + artificialVisc) * rho_i;
     

    }
       
}
void heating(float dt) {
    for (auto& b : bodies) {
        Vec3 v = b.vel;
        float speed2 = dot(v, v);
        float heat;

         heat = speed2*0.05f;
         
         heat += b.mass;
        b.heat += (heat * hmulti) * dt;
    }
}
void starcolision() {
    for (int i = 0; i < bodies.size(); i++)
    {
        if (!bodies[i].iscenter) continue;   // STAR

        Vec3 starPos = bodies[i].pos;
        float starRadius = bodies[i].size;

        for (int j = 0; j < bodies.size(); j++)
        {
            if (j == i) continue;
            

            Vec3 r = bodies[j].pos - starPos;
            float dist = length(r);

            float minDist = starRadius + bodies[j].size;

           
            if (dist < minDist)
            {
                Vec3 n = r / (dist + 1e-6f);

                // snap to surface (like AABB clamp)
                bodies[j].pos = starPos + n * minDist;

                // remove inward velocity (reflect-less)
                float vn = dot(bodies[j].vel, n);
                if (vn < 0.0f)
                    bodies[j].vel -= n * vn;


            }
        }
    }
}
void stepsph(float subDt) {

    buildgrid(bodies);
    computedensity();
    coputepressureforce(subDt);

    //perf eaters


}
void updateBodies(float dt) {
    //slowass intigration
    const float timeStep = dt;
    float heatDecay = cold * dt;
    Vec3 acc_new;
   
    for (int i = 0; i < bodies.size(); i++) {
        Body& b = bodies[i];
      
        b.acc_old = b.acl;
        if (lockstar == true) {
            if (b.iscenter)b.vel = Vec3(0);
        }
        b.pos += b.vel * dt + 0.5f * b.acc_old * dt * dt;

        b.acl = Vec3(0);
        b.force = Vec3(0);

    }
    
       
    
    if (gravityFun) {
        gravity();
    }
    if (colisionFun) {
        stepsph(dt);
    }

    for (int i = 0; i < (int)bodies.size(); ++i) {
        Body& b = bodies[i];
       
        if (b.iscenter)
            acc_new = b.acl + b.force / b.mass;
        else
            acc_new = b.acl + b.force / b.mass;

        b.vel += 0.5f * (b.acc_old + acc_new) * dt;
        b.acl = acc_new;
      b.heat *= exp(-cold * dt);
      b.heat = std::clamp(b.heat, 0.0f, MAX_HEAT);
    
        if (b.pos.x > 5000 || b.pos.x < -5000 || b.pos.y>3000 || b.pos.y < -3000 || b.pos.z > 5000 || b.pos.z < -5000) {
            b.alive = false;
            
        }
       
    }
     
   
}
void updatePhysics(float dt) {
    float subDt = dt / (float)substeps;
   

    for (int step = 0; step < substeps; step++) {

        for (int i = 0; i < bodies.size(); i++) {
            bodies[i].force = { 0, 0, 0 };
            bodies[i].acl = { 0, 0, 0 };
        }
       
        if (gravityFun==true) {
            gravity();
        }
       

        

        if (colisionFun == true) {
           
            stepsph(subDt);
           
        }
        if (heateffect == true) {
            heating(subDt);
        }
       
        
       
        if (updateFun==true) {
            updateBodies(subDt);
        }
      
            starcolision();
        
    }

    // 5. Heat decay (cooling over time)

    //org-red-org
    float coolingRate = 0.5f;  // Adjust as needed
    for (Body& b : bodies) {
        b.heat *= std::exp(-coolingRate * dt);
        b.heat = std::clamp(b.heat, 0.0f, MAX_HEAT);
        float t = std::clamp(b.heat / MAX_HEAT, 0.0f, 1.0f);

       
        t = std::pow(t, 0.95f);

        // base color (stored once at spawn)
        float br = (float)b.br;
        float bg = (float)b.bg;
        float bb = (float)b.bb;

        // fade base → pure red
        float r = std::lerp(br, 255.0f, t);
        float g = std::lerp(bg, 0.0f, t);
        float bl = std::lerp(bb, 0.0f, t);

        b.r = std::clamp((int)r, 0, 255);
        b.g = std::clamp((int)g, 0, 255);
        b.b = std::clamp((int)bl, 0, 255);



    }
}
void drawAll() {
    
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND); // IMPORTANT: no transparency
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const int VERTS_PER_BODY = 3;
    size_t totalVerts = bodies.size() * VERTS_PER_BODY;

    ensureVBOCapacity(totalVerts);
    ensureIBOCapacity(bodies.size());

    std::vector<GLVertex> verts(totalVerts);

    for (int i = 0; i < bodies.size(); i++) {
        const Body& b = bodies[i];

        float cr = b.r / 255.0f;
        float cg = b.g / 255.0f;
        float cb = b.b / 255.0f;

        int v = i * 3;

        // Fullscreen-covering triangle offsets
        const float ox[3] = { -1.0f,  3.0f, -1.0f };
        const float oy[3] = { -1.0f, -1.0f,  3.0f };

        for (int k = 0; k < 3; k++) {
            verts[v + k].px = b.pos.x;
            verts[v + k].py = b.pos.y;
            verts[v + k].pz = b.pos.z;

            verts[v + k].radius = b.size;

            verts[v + k].cr = cr;
            verts[v + k].cg = cg;
            verts[v + k].cb = cb;
            verts[v + k].ca = 1.0f;

            verts[v + k].ox = ox[k];
            verts[v + k].oy = oy[k];

           
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        verts.size() * sizeof(GLVertex),
        verts.data()
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(program);

    glm::mat4 proj = glm::perspective(
        glm::radians(camera.fov),
        (float)screenWidth / (float)screenHeight,
        0.1f,
        20000.0f
    );

    glm::mat4 view = glm::lookAt(
        camera.position,
        camera.position + camera.forward,
        camera.up
    );

    glUniformMatrix4fv(
        glGetUniformLocation(program, "uProj"),
        1, GL_FALSE, glm::value_ptr(proj)
    );

    glUniformMatrix4fv(
        glGetUniformLocation(program, "uView"),
        1, GL_FALSE, glm::value_ptr(view)
    );
    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.6f, 1.0f));

glUniform3f(
    glGetUniformLocation(program, "uLightDir"),
    lightDir.x,
    lightDir.y,
    lightDir.z
);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, bodies.size() * 3);
    glBindVertexArray(0);

    glUseProgram(0);
}
void MaxFps(double avgMs) {
    static int cooldown = 0;
    static int direction = 0; // +1 increasing, -1 decreasing

    if (cooldown > 0) {
        cooldown--;
        return;
    }

    const double LOW = 15.5;
    const double HIGH = 17.5;

    if (avgMs < LOW) {
        totalBodies += 5;
        direction = +1;
        cooldown = 2; // wait 2 seconds
    }
    else if (avgMs > HIGH) {
        totalBodies -= 5;
        totalBodies = std::max(totalBodies, 10);
        direction = -1;
        cooldown = 2;
    }
    restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);

    
}
void updateCameraVectors(Camera& cam)
{
    float yawRad = glm::radians(cam.yaw);
    float pitchRad = glm::radians(cam.pitch);

    // Z-up world
    cam.forward = glm::normalize(glm::vec3(
        cos(yawRad) * cos(pitchRad),
        sin(yawRad) * cos(pitchRad),
        sin(pitchRad)
    ));

    cam.right = glm::normalize(glm::cross(cam.forward, glm::vec3(0, 0, 1)));
    cam.up = glm::normalize(glm::cross(cam.right, cam.forward));
}
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    // Only rotate when holding left click
    if (!cameraRotating) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        return;
    }

    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
    }

    float dx = (float)(xpos - lastMouseX);
    float dy = (float)(lastMouseY - ypos);

    lastMouseX = xpos;
    lastMouseY = ypos;

    dx *= mouseSensitivity;
    dy *= mouseSensitivity;

    camera.yaw -= dx;
    camera.pitch += dy;

    if (camera.pitch > 89.0f)  camera.pitch = 89.0f;
    if (camera.pitch < -89.0f) camera.pitch = -89.0f;

    updateCameraVectors(camera);
}
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            cameraRotating = true;

            // 🔴 HARD RESET mouse origin
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
            firstMouse = false;
        }
        else if (action == GLFW_RELEASE) {
            cameraRotating = false;
            firstMouse = true; // prepare for next drag
        }
    }
}
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    camera.fov -= (float)yoffset * scrollSensitivity;

    if (camera.fov < 15.0f)  camera.fov = 15.0f;
    if (camera.fov > 120.0f) camera.fov = 120.0f;
}
void updateCameraMovement(GLFWwindow* window, float dt) {

    float speed = 250.0f * dt;  // tweak this

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.position += camera.forward * speed;

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.position -= camera.forward * speed;

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.position -= camera.right * speed;

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.position += camera.right * speed;

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.position += camera.up * speed;

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera.position -= camera.up * speed;

   
}
void buttons(GLFWwindow* window) {

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
}
void framebuffer_size_callback(GLFWwindow* w, int width, int height) {
    glViewport(0, 0, width, height);
}
int main() {
    srand((unsigned)time(nullptr));
    // CHANGE: Initialize GLFW instead of SFML
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n"; return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);


    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "N-Body Simulation - OpenGL", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    if (!window) { std::cerr << "Failed create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    // CHANGE: Initialize GLAD instead of SFML GL
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n"; return -1;
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glViewport(0, 0, screenWidth, screenHeight);

    // CHANGE: Setup callbacks before ImGui
    updateCameraVectors(camera);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glfwSetScrollCallback(window, scrollCallback);

    // CHANGE: Initialize ImGui for GLFW+OpenGL3
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    const char* glsl_version = "#version 330";
    ImGui_ImplOpenGL3_Init(glsl_version);

    // CHANGE: Create shader program
    program = createProgram(vertexShaderSource, fragmentShaderSource);
    ensureVBOCapacity(1024);

    

    float accumulator = 0.f;
    float fps = 0.f, avgFps = 0.f, maxFps = 0.f, minFps = 9999.f;
    float fpsTimer = 0.f;
    int fpsCount = 0;

    
    const float targetFPS = 60.0f;
    const float upperThreshold = 65.0f;
    const float lowerThreshold = 55.0f;

    registerBody(totalBodies, size, setMass, radius, orbitalspeed);

    
    bool option3 = false;
   
    double lastTime = glfwGetTime();
    double fpsClock = lastTime;

    view.cx = screenWidth * 0.5f;
    view.cy = screenHeight * 0.5f;
    view.height = (float)screenHeight;
    view.aspect = (float)screenWidth / (float)screenHeight;
    view.zoom = 1.0f;

    while (!glfwWindowShouldClose(window)) {
        
        glfwPollEvents();

       
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // UI 
        ImGui::Begin("Settings");
        if (totalBodies != bodies.size()) {
            ImGui::Text("BODY COUNT MISMATCH");
        }
        ImGui::Text("FPS: %.0f (Min: %.0f / Max: %.0f)", avgFps, minFps, maxFps);
        ImGui::Text("physics: %5.3f ms", fuc_ms);
        ImGui::Text("yaw: %.00f  pitch: %.00f", y, p);
        ImGui::Text("x:%.0f y %.0f z %.0f", wx,wy,wz);
       
        ImGui::Text("variables");
        ImGui::SliderFloat("speed", &simspeed, 0.001f, 10.0f);
       
       
        ImGui::SliderFloat("cold", &cold, 0.0f, 20.0f);
        ImGui::SliderFloat("heat", &hmulti, 0.0f, 20.0f);
        ImGui::SliderFloat("conduction rate", &cr, 0.0f, 10.0f);
        ImGui::Checkbox("heat effect", &heateffect);
        ImGui::SliderFloat("smoothing", &h,0.0f,20.0f);
         
        ImGui::SliderFloat("rest density", &rest_density,0.001f,10.0f);
        ImGui::SliderFloat("pressure f", &pressure,0.01f,10000.0f);
      
        ImGui::InputFloat("alpha visc", &alpha_visc);
        ImGui::InputFloat("beta visc", &beta_visc);
        ImGui::InputFloat("Gamma", &gamma);
        ImGui::InputFloat("G", &G);
       
        ImGui::Text("material settings");
        ImGui::InputInt("Total Bodies", &totalBodies);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
           
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        if (ImGui::Button("restart sim")) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::InputInt("x", &nx);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::InputInt("y", &ny);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::InputInt("Z", &nz);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::InputFloat("size", &size);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            starsize = size * 3;
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::InputFloat("mass", &setMass);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
      

        ImGui::Text("star settings");
        ImGui::InputFloat("star mass", &centermass);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            for (int i = 0; i < totalBodies; ++i) {
                if (bodies[i].iscenter == true) bodies[i].mass = centermass;
            }
        }
        ImGui::InputFloat("star size", &starsize);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            for (int i = 0; i < totalBodies; ++i) {
                if (bodies[i].iscenter == true) bodies[i].size = starsize;
            }
        }
      
        ImGui::InputFloat("ring radius", &radius);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::InputFloat("ring max radius", &Maxradius);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::InputFloat("orbital speed", &orbitalspeed);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::Checkbox("lock star position", &lockstar);

        ImGui::Text("physics");
        ImGui::Checkbox("collision", &colisionFun);
        ImGui::Checkbox("gravity", &gravityFun);
        ImGui::Checkbox("update bodies", &updateFun);
        ImGui::Checkbox("max at 60 fps", &option3);
        
       

        ImGui::Text("formation");
        ImGui::RadioButton("orbital ", &mode, 0);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
       
        ImGui::RadioButton("random ", &mode, 1);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        if (mode == 1) {
            ImGui::InputInt("box size", &maxX);
            maxY = maxX;
        }
   
       
        ImGui::RadioButton("double galaxy ", &mode, 2);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::RadioButton("sphere ", &mode, 3);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        ImGui::RadioButton("earth and theia ", &mode, 4);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            restartSimulation(totalBodies, size, setMass, radius, orbitalspeed);
        }
        if (mode == 4) {
            ImGui::InputFloat("impact speed", &impactSpeed);
            ImGui::InputFloat("y speed", &yspeed);
            ImGui::InputFloat("distance", &DST);
        }
      
      
       
         
        
        
        ImGui::Text("performance");
       

        

        ImGui::Text("physics: %5.3f ms", fuc_ms);
        ImGui::InputInt("substeps", &substeps);
        ImGui::Checkbox("apply force omp", &omp1);
        ImGui::Checkbox(" omp2", &omp2);
        ImGui::Checkbox(" omp3", &omp3);
        ImGui::Checkbox(" omp4", &omp4);
       
       
        ImGui::InputInt("c1 ", &cc1);
        
        ImGui::End();
        // Timing
        double now = glfwGetTime();
        double frameTime = now - lastTime;
        lastTime = now;
        accumulator += (float)frameTime;
        float dt = (float)frameTime;

       
        updateCameraMovement(window,dt);
        buttons(window);
        y = camera.yaw;
        p = camera.pitch;
        wx = camera.position.x;
        wy = camera.position.y;
        wz = camera.position.z;
        if (mode != 5) {
            gravityFun = true;
        }
        float effectiveDt = fixedDt * simspeed;
        auto t0 = std::chrono::high_resolution_clock::now();
        while (accumulator >= fixedDt) {
            

            
            updatePhysics(effectiveDt);
       
            
            
            
            accumulator -= fixedDt;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        fuc_ms_avg += ms;
        fuc_samples++;
        if (fuc_samples >= 60) {   // 1 second @ 60 FPS
            fuc_ms = fuc_ms_avg / fuc_samples;
            fuc_ms_avg = 0.0;
            fuc_samples = 0;
        }
        if (option3) {
            MaxFps(fuc_ms);
        }
        bodies.erase(
            std::remove_if(bodies.begin(), bodies.end(),
                [](const Body& b) { return !b.alive; }),
            bodies.end());

        // CHANGE: OpenGL rendering instead of SFML
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
       
       
       
        drawAll();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // FPS measurement
        double elapsed = glfwGetTime() - fpsClock;
        fpsClock = glfwGetTime();
        fps = (elapsed > 0.0) ? 1.0 / elapsed : fps;
        fpsTimer += (float)elapsed;
        fpsCount++;
        if (fps > maxFps) maxFps = (float)fps;
        if (fps < minFps) minFps = (float)fps;
        if (fpsTimer >= 0.5f) {
            avgFps = fpsCount / fpsTimer;
            fpsTimer = 0.f;
            fpsCount = 0;
        }
    }

    // CHANGE: Cleanup OpenGL resources
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (program) glDeleteProgram(program);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ibo) glDeleteBuffers(1, &ibo);
    if (vao) glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}