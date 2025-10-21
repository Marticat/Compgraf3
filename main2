
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>

// struct
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

struct TriIdx {
    int a, b, c;
};

// global
std::vector<glm::vec3> baseVerts;     
std::vector<TriIdx> baseTris;        
std::vector<Vertex> vertexBuffer;      
std::vector<unsigned int> indexBuffer; 

GLuint VAO = 0, VBO = 0, EBO = 0, shaderProgram = 0;
bool perspectiveProj = true;
bool useGouraud = false; //false -> Phong , true -> Gouraud

// Camera 
float camTheta = 0.0f;
float camRadius = 5.0f;
float camHeight = 0.0f;

// Light1 (object-space) on cylinder (user-controlled)
float light1Theta = 0.0f;
float light1Radius = 5.0f;
float light1Height = 1.0f;

// For toggles and state
GLFWwindow* g_window = nullptr;

// Material 
struct Material {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
};

std::map<int, Material> materials;
int currentMaterial = 1;

// Light struct for GLSL uniforms
struct Light {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
};

//Shaders (combined Gouraud/Phong via uniform useGouraud)
const char* vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

// light/material uniforms
uniform vec3 light1Pos_world; // object-space light position transformed in app to world (or model)
uniform vec3 light2Pos_view;  // camera-space light position (in view coords)
uniform vec3 lightAmbient[2];
uniform vec3 lightDiffuse[2];
uniform vec3 lightSpecular[2];

uniform vec3 materialAmbient;
uniform vec3 materialDiffuse;
uniform vec3 materialSpecular;
uniform float materialShininess;

uniform bool useGouraud; // if true compute lighting in vertex shader

out vec3 vPos_view;
out vec3 vNormal_view;
out vec3 gouraudColor;

vec3 computePhong(vec3 pos_view, vec3 normal_view) {
    vec3 viewDir = normalize(-pos_view); // eye at (0,0,0) in view space
    vec3 result = vec3(0.0);
    // light1: transform from world to view done in app -> passed as view-space? We'll pass both in view space.
    vec3 lightPos0 = (view * vec4(light1Pos_world, 1.0)).xyz; // but view not allowed here — so better to pass light1 already in view in uniform
    // NOTE: we will not use this path; instead app will pass both lights in view space as uniforms.
    return result;
}

void main() {
    // transform
    vec4 pos_world = model * vec4(aPos, 1.0);
    vec4 pos_view4 = view * pos_world;
    vec3 pos_view = pos_view4.xyz;
    vPos_view = pos_view;

    vec3 normal_world = normalize(normalMatrix * aNormal);
    vec3 normal_view = normalize((mat3(view) * normal_world));
    vNormal_view = normal_view;

    gl_Position = projection * pos_view4;

    if (useGouraud) {
        // compute lighting per-vertex in view space using two lights passed in view space
        vec3 ambientSum = vec3(0.0);
        vec3 diffuseSum = vec3(0.0);
        vec3 specSum = vec3(0.0);

        // Light 0 (object-space light passed transformed to view in uniform light1Pos_view)
        for (int i = 0; i < 2; ++i) {
            // We'll access light uniforms via arrays using manual mapping in app (two separate uniforms)
        }
        // Because GLSL arrays with dynamic indices are inconvenient here, we'll directly use two sets:
        // (we'll use separate uniforms in the program; to keep vertex shader short, replicate)
    }
    // if not useGouraud, gouraudColor remains unused
}
)";


const char* gouraudVertex = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

// Lights (in VIEW space)
uniform vec3 lightPos_view0;
uniform vec3 lightAmbient0;
uniform vec3 lightDiffuse0;
uniform vec3 lightSpecular0;

uniform vec3 lightPos_view1;
uniform vec3 lightAmbient1;
uniform vec3 lightDiffuse1;
uniform vec3 lightSpecular1;

// Material
uniform vec3 materialAmbient;
uniform vec3 materialDiffuse;
uniform vec3 materialSpecular;
uniform float materialShininess;

out vec3 vertColor; // interpolated across triangle

vec3 computeLight(vec3 pos_view, vec3 normal_view, vec3 lightPos_view, vec3 La, vec3 Ld, vec3 Ls) {
    vec3 ambient = La * materialAmbient;
    vec3 Ldir = normalize(lightPos_view - pos_view);
    float diff = max(dot(normal_view, Ldir), 0.0);
    vec3 diffuse = Ld * (materialDiffuse * diff);
    vec3 spec = vec3(0.0);
    if (diff > 0.0) {
        vec3 viewDir = normalize(-pos_view);
        vec3 reflectDir = reflect(-Ldir, normal_view);
        float specFactor = pow(max(dot(viewDir, reflectDir), 0.0), materialShininess);
        spec = Ls * (materialSpecular * specFactor);
    }
    return ambient + diffuse + spec;
}

void main() {
    vec4 pos_world = model * vec4(aPos, 1.0);
    vec4 pos_view4 = view * pos_world;
    vec3 pos_view = pos_view4.xyz;

    vec3 normal_world = normalize(normalMatrix * aNormal);
    vec3 normal_view = normalize((mat3(view) * normal_world));

    vec3 color0 = computeLight(pos_view, normal_view, lightPos_view0, lightAmbient0, lightDiffuse0, lightSpecular0);
    vec3 color1 = computeLight(pos_view, normal_view, lightPos_view1, lightAmbient1, lightDiffuse1, lightSpecular1);

    vertColor = color0 + color1;

    gl_Position = projection * pos_view4;
}
)";

const char* gouraudFragment = R"(
#version 330 core
in vec3 vertColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vertColor, 1.0);
}
)";

const char* phongVertex = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

out vec3 vPos_view;
out vec3 vNormal_view;

void main() {
    vec4 pos_world = model * vec4(aPos, 1.0);
    vec4 pos_view4 = view * pos_world;
    vPos_view = pos_view4.xyz;
    vec3 normal_world = normalize(normalMatrix * aNormal);
    vNormal_view = normalize(mat3(view) * normal_world);
    gl_Position = projection * pos_view4;
}
)";

const char* phongFragment = R"(
#version 330 core
in vec3 vPos_view;
in vec3 vNormal_view;
out vec4 FragColor;

uniform vec3 lightPos_view0;
uniform vec3 lightAmbient0;
uniform vec3 lightDiffuse0;
uniform vec3 lightSpecular0;

uniform vec3 lightPos_view1;
uniform vec3 lightAmbient1;
uniform vec3 lightDiffuse1;
uniform vec3 lightSpecular1;

uniform vec3 materialAmbient;
uniform vec3 materialDiffuse;
uniform vec3 materialSpecular;
uniform float materialShininess;

vec3 computeLight(vec3 pos_view, vec3 normal_view, vec3 lightPos_view, vec3 La, vec3 Ld, vec3 Ls) {
    vec3 ambient = La * materialAmbient;
    vec3 Ldir = normalize(lightPos_view - pos_view);
    float diff = max(dot(normal_view, Ldir), 0.0);
    vec3 diffuse = Ld * (materialDiffuse * diff);
    vec3 spec = vec3(0.0);
    if (diff > 0.0) {
        vec3 viewDir = normalize(-pos_view);
        vec3 reflectDir = reflect(-Ldir, normal_view);
        float specFactor = pow(max(dot(viewDir, reflectDir), 0.0), materialShininess);
        spec = Ls * (materialSpecular * specFactor);
    }
    return ambient + diffuse + spec;
}

void main() {
    vec3 n = normalize(vNormal_view);
    vec3 c0 = computeLight(vPos_view, n, lightPos_view0, lightAmbient0, lightDiffuse0, lightSpecular0);
    vec3 c1 = computeLight(vPos_view, n, lightPos_view1, lightAmbient1, lightDiffuse1, lightSpecular1);
    vec3 col = c0 + c1;
    FragColor = vec4(col, 1.0);
}
)";


GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char info[1024]; glGetShaderInfoLog(s, 1024, nullptr, info);
        std::cerr << "Shader compile error: " << info << std::endl;
    }
    return s;
}
GLuint linkProgram(GLuint v, GLuint f) {
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char info[1024]; glGetProgramInfoLog(p, 1024, nullptr, info);
        std::cerr << "Program link error: " << info << std::endl;
    }
    return p;
}

GLuint gouraudProgram = 0;
GLuint phongProgram = 0;

bool loadSMF(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Cannot open: " << filename << std::endl;
        return false;
    }
    baseVerts.clear();
    baseTris.clear();

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string t; iss >> t;
        if (t == "v") {
            float x, y, z; if (iss >> x >> y >> z) baseVerts.push_back(glm::vec3(x, y, z));
        }
        else if (t == "f") {
            int a, b, c; if (iss >> a >> b >> c) {
                baseTris.push_back({ a - 1,b - 1,c - 1 });
            }
        }
    }

    if (baseTris.empty() || baseVerts.empty()) return false;

    std::vector<glm::vec3> normals(baseVerts.size(), glm::vec3(0.0f));
    for (auto& t : baseTris) {
        glm::vec3 v1 = baseVerts[t.a];
        glm::vec3 v2 = baseVerts[t.b];
        glm::vec3 v3 = baseVerts[t.c];
        glm::vec3 faceN = glm::normalize(glm::cross(v2 - v1, v3 - v1));
        normals[t.a] += faceN;
        normals[t.b] += faceN;
        normals[t.c] += faceN;
    }
    for (size_t i = 0; i < normals.size(); ++i) normals[i] = glm::normalize(normals[i]);

    vertexBuffer.clear();
    indexBuffer.clear();
    vertexBuffer.reserve(baseVerts.size());
    for (size_t i = 0; i < baseVerts.size(); ++i) {
        vertexBuffer.push_back({ baseVerts[i], normals[i] });
    }
    indexBuffer.reserve(baseTris.size() * 3);
    for (auto& t : baseTris) {
        indexBuffer.push_back(t.a);
        indexBuffer.push_back(t.b);
        indexBuffer.push_back(t.c);
    }
    return true;
}

glm::vec3 centerModel() {
    if (vertexBuffer.empty()) return glm::vec3(0.0f);
    glm::vec3 c(0.0f);
    for (auto& v : vertexBuffer) c += v.position;
    c /= (float)vertexBuffer.size();
    for (auto& v : vertexBuffer) v.position -= c;
    return c;
}

void setupGLBuffers() {
    if (VAO) {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
    }
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertexBuffer.size() * sizeof(Vertex), vertexBuffer.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.size() * sizeof(unsigned int), indexBuffer.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void framebuffer_size_callback(GLFWwindow* window, int w, int h) {
    if (h == 0) return;
    glViewport(0, 0, w, h);
}

void processInput(GLFWwindow* window) {
    const float angleSpeed = 0.03f;
    const float heightSpeed = 0.05f;
    const float zoomSpeed = 0.05f;

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camTheta -= angleSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camTheta += angleSpeed;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camHeight += heightSpeed;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camHeight -= heightSpeed;

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) camRadius += zoomSpeed;
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) camRadius = std::max(0.1f, camRadius - zoomSpeed);

    static bool tPrev = false;
    bool tNow = glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS;
    if (tNow && !tPrev) {
        useGouraud = !useGouraud;
        std::cout << (useGouraud ? "[Gouraud]" : "[Phong]") << " shading selected\n";
    }
    tPrev = tNow;

    static bool onePrev = false, twoPrev = false, threePrev = false;
    bool oneNow = glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS;
    bool twoNow = glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS;
    bool threeNow = glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS;
    if (oneNow && !onePrev) { currentMaterial = 1; std::cout << "Material 1 selected\n"; }
    if (twoNow && !twoPrev) { currentMaterial = 2; std::cout << "Material 2 selected\n"; }
    if (threeNow && !threePrev) { currentMaterial = 3; std::cout << "Material 3 selected\n"; }
    onePrev = oneNow; twoPrev = twoNow; threePrev = threeNow;

    // move light1 on cylinder: Z/X angle-,angle+; C/V radius-,radius+; B/N height-,height+
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) light1Theta -= 0.03f;
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) light1Theta += 0.03f;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) light1Radius = std::max(0.1f, light1Radius - 0.05f);
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) light1Radius += 0.05f;
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) light1Height -= 0.05f;
    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) light1Height += 0.05f;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Toggle between perspective and orthographic projection
    static bool pPrev = false;
    bool pNow = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
    if (pNow && !pPrev) {
        perspectiveProj = !perspectiveProj;
        std::cout << (perspectiveProj ? "Perspective projection\n" : "Orthographic projection\n");
    }
    pPrev = pNow;
}

//Program creation
void createPrograms() {
    // Gouraud
    GLuint gv = compileShader(GL_VERTEX_SHADER, gouraudVertex);
    GLuint gf = compileShader(GL_FRAGMENT_SHADER, gouraudFragment);
    gouraudProgram = linkProgram(gv, gf);
    glDeleteShader(gv); glDeleteShader(gf);

    // Phong
    GLuint pv = compileShader(GL_VERTEX_SHADER, phongVertex);
    GLuint pf = compileShader(GL_FRAGMENT_SHADER, phongFragment);
    phongProgram = linkProgram(pv, pf);
    glDeleteShader(pv); glDeleteShader(pf);
}

//Render
void renderScene(int width, int height) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float camX = camRadius * cos(camTheta);
    float camY = camRadius * sin(camTheta);
    float camZ = camHeight;

    glm::mat4 view = glm::lookAt(glm::vec3(camX, camY, camZ), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 proj = perspectiveProj ? glm::perspective(glm::radians(60.0f), (float)width / height, 0.1f, 100.0f)
        : glm::ortho(-5.f, 5.f, -5.f * (float)height / width, 5.f * (float)height / width, 0.1f, 100.0f);
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));


    // Light 0: object-space cylinder (compute in world coordinates; model is identity but we treat as model)
    glm::vec3 light1_world = glm::vec3(light1Radius * cos(light1Theta), light1Radius * sin(light1Theta), light1Height);
    glm::vec3 light1_view = glm::vec3(view * glm::vec4(light1_world, 1.0f));

    // Light 1: camera-space light near eye -> put it slightly in front of camera in view space, e.g. at (0,0,1)
    glm::vec3 light2_view = glm::vec3(0.0f, 0.0f, 1.0f);

    // Light properties (same for both lights for simplicity; can be distinct)
    glm::vec3 lightAmbient0(0.2f, 0.2f, 0.2f);
    glm::vec3 lightDiffuse0(0.6f, 0.6f, 0.6f);
    glm::vec3 lightSpec0(1.0f, 1.0f, 1.0f);

    glm::vec3 lightAmbient1(0.15f, 0.15f, 0.15f);
    glm::vec3 lightDiffuse1(0.5f, 0.5f, 0.5f);
    glm::vec3 lightSpec1(1.0f, 1.0f, 1.0f);

    // choose program
    GLuint prog = useGouraud ? gouraudProgram : phongProgram;
    glUseProgram(prog);

    // set common uniforms
    GLint locModel = glGetUniformLocation(prog, "model");
    GLint locView = glGetUniformLocation(prog, "view");
    GLint locProj = glGetUniformLocation(prog, "projection");
    GLint locNormal = glGetUniformLocation(prog, "normalMatrix");
    glUniformMatrix4fv(locModel, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(locView, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(locProj, 1, GL_FALSE, glm::value_ptr(proj));
    glUniformMatrix3fv(locNormal, 1, GL_FALSE, glm::value_ptr(normalMatrix));

    // pass lights (view-space positions)
    GLint uLp0 = glGetUniformLocation(prog, "lightPos_view0");
    GLint uLa0 = glGetUniformLocation(prog, "lightAmbient0");
    GLint uLd0 = glGetUniformLocation(prog, "lightDiffuse0");
    GLint uLs0 = glGetUniformLocation(prog, "lightSpecular0");

    GLint uLp1 = glGetUniformLocation(prog, "lightPos_view1");
    GLint uLa1 = glGetUniformLocation(prog, "lightAmbient1");
    GLint uLd1 = glGetUniformLocation(prog, "lightDiffuse1");
    GLint uLs1 = glGetUniformLocation(prog, "lightSpecular1");

    glUniform3fv(uLp0, 1, glm::value_ptr(light1_view));
    glUniform3fv(uLa0, 1, glm::value_ptr(lightAmbient0));
    glUniform3fv(uLd0, 1, glm::value_ptr(lightDiffuse0));
    glUniform3fv(uLs0, 1, glm::value_ptr(lightSpec0));

    glUniform3fv(uLp1, 1, glm::value_ptr(light2_view));
    glUniform3fv(uLa1, 1, glm::value_ptr(lightAmbient1));
    glUniform3fv(uLd1, 1, glm::value_ptr(lightDiffuse1));
    glUniform3fv(uLs1, 1, glm::value_ptr(lightSpec1));

    // pass material 
    Material m = materials[currentMaterial];
    GLint ma = glGetUniformLocation(prog, "materialAmbient");
    GLint md = glGetUniformLocation(prog, "materialDiffuse");
    GLint ms = glGetUniformLocation(prog, "materialSpecular");
    GLint msh = glGetUniformLocation(prog, "materialShininess");
    glUniform3fv(ma, 1, glm::value_ptr(m.ambient));
    glUniform3fv(md, 1, glm::value_ptr(m.diffuse));
    glUniform3fv(ms, 1, glm::value_ptr(m.specular));
    glUniform1f(msh, m.shininess);

    // draw
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, (GLsizei)indexBuffer.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

//Main
int main(int argc, char** argv) {
    std::string modelPath;
    if (argc >= 2) modelPath = argv[1];
    else {
        modelPath = "C:/Users/Ad/Desktop/AdilzhanKurmetSE2320/models/sprtrd.smf";
        std::cout << "No model arg. Using default: " << modelPath << std::endl;
    }

    if (!loadSMF(modelPath)) {
        std::cerr << "Failed to load model\n";
        return -1;
    }

    size_t triCount = baseTris.size();
    std::cout << "Triangles loaded: " << triCount << std::endl;
    if (triCount < 100) std::cout << "⚠ Warning: less than 100 triangles\n";

    glm::vec3 shift = centerModel();
    std::cout << "Model centered. centroid shift: " << shift.x << "," << shift.y << "," << shift.z << "\n";

    // materials: define three distinct materials
    materials[1] = { glm::vec3(0.2f,0.2f,0.2f), glm::vec3(0.6f,0.6f,0.6f), glm::vec3(0.2f,0.2f,0.2f), 16.0f }; // dull grey
    // material 2 must match problem statement exactly:
    materials[2] = { glm::vec3(0.6f,0.2f,0.2f), glm::vec3(0.9f,0.1f,0.1f), glm::vec3(0.8f,0.8f,0.8f), 80.0f };
    materials[3] = { glm::vec3(0.05f,0.05f,0.1f), glm::vec3(0.3f,0.4f,0.9f), glm::vec3(1.0f,1.0f,1.0f), 32.0f }; // shiny bluish

    currentMaterial = 2; // start with material 2

    // GLFW init
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    g_window = glfwCreateWindow(900, 700, "Part2", NULL, NULL);
    if (!g_window) { std::cerr << "Window creation failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(g_window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n"; return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.12f, 0.12f, 0.12f, 1.0f);

    glfwSetFramebufferSizeCallback(g_window, framebuffer_size_callback);

    // create programs
    createPrograms();

    // build buffers
    setupGLBuffers();

    std::cout << "Controls:\n";
    std::cout << "A D : rotate camera\nW S : raise/lower camera\nR/F : zoom out/in\nP : toggle projection\nT : toggle shading (Gouraud/Phong)\n1/2/3 : select material\nZ/X : light angle -/+  C/V : light radius -/+  B/N : light height -/+\n";

    // main loop
    while (!glfwWindowShouldClose(g_window)) {
        processInput(g_window);

        int w, h; glfwGetFramebufferSize(g_window, &w, &h);
        if (w == 0 || h == 0) { glfwPollEvents(); continue; }

        renderScene(w, h);

        glfwSwapBuffers(g_window);
        glfwPollEvents();
    }

    // cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(gouraudProgram);
    glDeleteProgram(phongProgram);

    glfwTerminate();
    return 0;
}
