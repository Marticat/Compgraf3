// main.cpp
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

// ======= Structures =======
struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

struct Triangle {
    int v1, v2, v3;
};

// ======= Globals =======
std::vector<Vertex> vertices;
std::vector<Triangle> triangles;
GLuint VAO = 0, VBO = 0, shaderProgram = 0;
bool perspectiveProj = true; // renamed for clarity

// Camera (cylindrical coordinates)
float camTheta = 0.0f;   // angle θ (radians)
float camRadius = 5.0f;  // radius R
float camHeight = 0.0f;  // height H

GLFWwindow* g_window = nullptr;

// ======= Shaders =======
// NOTE: use 'flat' qualifier to prevent interpolation (flat shading)
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 view;
uniform mat4 projection;

flat out vec3 vertexColor;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    vertexColor = aColor;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
flat in vec3 vertexColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vertexColor, 1.0);
}
)";

// ======= Load SMF model =======
bool loadSMF(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    std::vector<glm::vec3> verts;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type == "v") {
            float x, y, z;
            if (!(iss >> x >> y >> z)) continue;
            verts.push_back(glm::vec3(x, y, z));
        }
        else if (type == "f") {
            int a, b, c;
            if (!(iss >> a >> b >> c)) continue;
            // SMF indices are 1-based -> convert to 0-based
            triangles.push_back({ a - 1, b - 1, c - 1 });
        }
        // ignore other tokens/comments
    }

    // Build vertex list: for each triangle compute single normal and assign to its 3 vertices
    for (auto& tri : triangles) {
        if (tri.v1 < 0 || tri.v2 < 0 || tri.v3 < 0) continue;
        if (tri.v1 >= (int)verts.size() || tri.v2 >= (int)verts.size() || tri.v3 >= (int)verts.size()) continue;
        glm::vec3 v1 = verts[tri.v1];
        glm::vec3 v2 = verts[tri.v2];
        glm::vec3 v3 = verts[tri.v3];
        glm::vec3 normal = glm::normalize(glm::cross(v2 - v1, v3 - v1));
        normal = glm::abs(normal); // visualize normal as color
        vertices.push_back({ v1, normal });
        vertices.push_back({ v2, normal });
        vertices.push_back({ v3, normal });
    }
    return true;
}

// ======= Compute and center model =======
glm::vec3 computeAndCenterModel() {
    glm::vec3 c(0.0f);
    if (vertices.empty()) return c;
    for (auto& v : vertices) c += v.position;
    c /= (float)vertices.size();
    for (auto& v : vertices) v.position -= c;
    return c;
}

// ======= Compile shader =======
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, info);
        std::cerr << (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment")
            << " shader compile error:\n" << info << std::endl;
    }
    return shader;
}

// ======= Check program link =======
bool checkProgram(GLuint prog) {
    GLint success;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        char info[1024];
        glGetProgramInfoLog(prog, 1024, nullptr, info);
        std::cerr << "Shader program link error:\n" << info << std::endl;
        return false;
    }
    return true;
}

// ======= OpenGL setup =======
void initGL() {
    GLuint vShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vShader);
    glAttachShader(shaderProgram, fShader);
    glLinkProgram(shaderProgram);
    if (!checkProgram(shaderProgram)) {
        std::cerr << "Shader program failed to link." << std::endl;
    }
    glDeleteShader(vShader);
    glDeleteShader(fShader);

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));
    glEnableVertexAttribArray(1);

    // unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// ======= GLFW callbacks =======
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    if (height == 0) return;
    glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // toggle projection on press (not repeat)
    if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        perspectiveProj = !perspectiveProj;
    }
}

// ======= Keyboard input continuous (camera movement) =======
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
}

// ======= Render =======
void render(int width, int height) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float camX = camRadius * cos(camTheta);
    float camY = camRadius * sin(camTheta);
    float camZ = camHeight;

    glm::mat4 view = glm::lookAt(glm::vec3(camX, camY, camZ),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));

    glm::mat4 projection;
    if (perspectiveProj) {
        projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 100.0f);
    }
    else {
        // set ortho box sized relative to radius so model fits
        float orthoSize = std::max(5.0f, camRadius + 1.0f);
        projection = glm::ortho(-orthoSize, orthoSize, -orthoSize * (float)height / width, orthoSize * (float)height / width, 0.1f, 100.0f);
    }

    glUseProgram(shaderProgram);
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertices.size());
    glBindVertexArray(0);
}

// ======= main =======
int main(int argc, char** argv) {
    std::string modelPath;
    if (argc >= 2) {
        modelPath = argv[1];
        std::cout << "Loading model from argument: " << modelPath << std::endl;
    }
    else {
        modelPath = "C:/Users/Ad/Desktop/AdilzhanKurmetSE2320/models/sprtrd.smf";
        std::cout << "No argument given. Using default modelPath:\n" << modelPath << std::endl;
        std::cout << "Tip: run as: program.exe path/to/model.smf\n";
    }

    if (!loadSMF(modelPath)) {
        std::cerr << "Failed to load model.\n";
        return -1;
    }

    size_t triCount = triangles.size();
    std::cout << "Triangles loaded: " << triCount << std::endl;
    if (triCount < 100) std::cout << "⚠ Warning: less than 100 triangles (required >=100)\n";

    glm::vec3 centroidShift = computeAndCenterModel();
    std::cout << "Model centered. Centroid shift: ("
        << centroidShift.x << ", " << centroidShift.y << ", " << centroidShift.z << ")\n";

    // Init GLFW and window
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    g_window = glfwCreateWindow(800, 600, "Flat Shading (Cylindrical Camera)", NULL, NULL);
    if (!g_window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(g_window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    // register callbacks
    glfwSetFramebufferSizeCallback(g_window, framebuffer_size_callback);
    glfwSetKeyCallback(g_window, key_callback);

    // GL init
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.12f, 0.12f, 0.12f, 1.0f);

    initGL();

    std::cout << "Controls:\n";
    std::cout << "A D  : rotate around model (angle θ)\n";
    std::cout << "W S  : move camera up/down (height H)\n";
    std::cout << "R / F: zoom out / in (radius R)\n";
    std::cout << "P     : toggle projection (perspective / orthographic)\n";

    while (!glfwWindowShouldClose(g_window)) {
        processInput(g_window);
        int w, h;
        glfwGetFramebufferSize(g_window, &w, &h);
        render(w == 0 ? 800 : w, h == 0 ? 600 : h);
        glfwSwapBuffers(g_window);
        glfwPollEvents();
    }

    // cleanup
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
