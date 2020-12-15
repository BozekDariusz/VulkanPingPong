#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>


#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <set>
#include<string>


#include <stdlib.h>  
#include <array>
#include <chrono>

const int WIDTH = 1920;
const int HEIGHT = 1080;

int Left = 0;
int Right = 0;
int start = 0;
bool reset = 0;
float EPSILON = 0.1f;


struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;


    //binding description tells vulkan at which rate to load data from memory throughout the vertices, it specifies the number of bytes between each entry and
    //whether to move to the next entry after each vertex or instance (instanced rendering - same object rendered many times but in different positions)
    static VkVertexInputBindingDescription getBindingDescription() {

        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    //description of attributes of a vertex
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);


        return attributeDescriptions;
    }


};


 std::vector<Vertex> vertices;

 std::vector<uint16_t> indices;


 struct UniformBufferObject {

     glm::mat4 view;
     glm::mat4 proj;

 };
 struct UniformBufferModelObject {

     glm::mat4 model;

 };

struct Mesh {
    //vertices and indices
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
    std::vector<VkDescriptorSet> descriptorSets;
    //uniform buffers storage
    std::vector<VkBuffer> uniformModelBuffers;
    //memory for the uniform buffers
    std::vector<VkDeviceMemory> uniformModelBuffersMemory;
    glm::vec3 scale;
    std::string name;
    UniformBufferModelObject ubo;
    glm::vec2 direction;//FIXHERE move to Ball 
    int width;
    int height;
    Mesh() {}

    Mesh(const std::vector<Vertex> v, const std::vector<uint16_t> i) :vertices(v), indices(i) {

        scale = glm::vec3(1.0f, 1.0f, 1.0f);
        ubo.model = glm::mat4(1);
    }  
  
    
};

struct Ball :public Mesh {

   

    Ball(const std::vector<Vertex> v, const std::vector<uint16_t> i, std::string  newName, glm::vec3 newScale) {

        scale = newScale;
        vertices = v;
        indices = i;
        ubo.model = glm::mat4(1);
        name = newName;
    }
};

struct Platform :public Mesh {


    Platform(const std::vector<Vertex> v, const std::vector<uint16_t> i, std::string  newName, glm::vec3 newScale) {

        scale = newScale;
        vertices = v;
        indices = i;
        ubo.model = glm::mat4(1);
        name = newName;
    }
};

struct Tile :public Mesh {

    bool isVisible = true;
    Tile(const std::vector<Vertex> v, const std::vector<uint16_t> i, std::string  newName, glm::vec3 newScale) {

        scale = newScale;
        vertices = v;
        indices = i;
        ubo.model = glm::mat4(1);
        name = newName;

    }
};

Platform platform({
     {{0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
     {{100.0f, 20.0f}, {1.0f, 1.0f, 1.0f}},
     {{100.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
     {{0.0f, 20.0f}, {1.0f, 1.0f, 1.0f}}
     },
     { 2, 1, 0 , 0 , 1 ,3 },"Platform", glm::vec3(1.0f, 0.5f, 0.0f));



Ball ball(
    { {{1.0f,1.0f,}, {1.0f, 1.0f, 1.0f}},
{{11.0f,1.0f,}, {1.0f, 1.0f, 1.0f}},
{{9.09017f,6.87785f,}, {1.0f, 1.0f, 1.0f}},
{{4.09017f,10.5106f,}, {1.0f, 1.0f, 1.0f}},
{{-2.09017f,10.5106f,}, {1.0f, 1.0f, 1.0f}},
{{-7.09017f,6.87785f,}, {1.0f, 1.0f, 1.0f}},
{{-9.0f,0.999999f,}, {1.0f, 1.0f, 1.0f}},
{{-7.09017f,-4.87785f,}, {1.0f, 1.0f, 1.0f}},
{{-2.09017f,-8.51056f,}, {1.0f, 1.0f, 1.0f}},
{{4.09017f,-8.51056f,}, {1.0f, 1.0f, 1.0f}},
{{9.09017f,-4.87785f,}, {1.0f, 1.0f, 1.0f}},
    },
    { 0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 7, 0, 7, 8, 0, 8, 9, 0, 9, 10, 0, 10, 1, },"Ball",glm::vec3(1.0f,1.0f,1.0f)
);


std::vector< Mesh*> allMeshes;
std::vector< Tile*> allTiles;


//number of frames processed concurrently
const int MAX_FRAMES_IN_FLIGHT = 2;

//specify validation layers to enable
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

//specify required device extensions
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//add configuration variables to the program to check whether to enable validation layers or not 
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

//function which creates object of debug messenger
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    //find address of vkCreateDebugUtilsMessengerEXT function and pass arguments to it
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    //if function could not be loaded throw extension function error
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

//clean up debug messenger object
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

//structure for queue families, use std::optional<> to determine if value(queue family) exists or not 
struct QueueFamilyIndices {
    //graphics queue family
    std::optional<uint32_t> graphicsFamily;
    //presentation queue family
    std::optional<uint32_t> presentFamily;

    //check if graphicsFamily and presentFamily values exist
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

/*
structure to pass details about swap chain which will include basic surface capabilities (min/max number of images in swap chain and min/max width/height of images),
surface formats(pixel format, colour space) and available presentation modes
*/
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

//wrapper class for:initializing GLFW and creating a window, storing Vulkan objects, initiating them, rendering frames and deallocating resources after quitting the main loop
class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    //instance of Vulkan library
    VkInstance instance;
    //handle for callback function
    VkDebugUtilsMessengerEXT debugMessenger;
    //surface for rendering
    VkSurfaceKHR surface;

    //GPU to be used
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    //logical device handle
    VkDevice device;

    //handle to the graphics queue
    VkQueue graphicsQueue;
    //handle to the presentation queue
    VkQueue presentQueue;

    //swap chain object
    VkSwapchainKHR swapChain;
    //swap chain images handles storage
    std::vector<VkImage> swapChainImages;
    //swap chain image format storage
    VkFormat swapChainImageFormat;
    //swap chain extent storage
    VkExtent2D swapChainExtent;
    //image views storage
    std::vector<VkImageView> swapChainImageViews;
    //framebuffers storage
    std::vector<VkFramebuffer> swapChainFramebuffers;

    //storage for render pass object
    VkRenderPass renderPass;
    //all of the descriptor bindings are combined into single VkDescriptorSetLayout
    VkDescriptorSetLayout descriptorSetLayout[2];
    //storage for pipeline layout object
    VkPipelineLayout pipelineLayout;
    //storage for graphics pipeline
    VkPipeline graphicsPipeline;

    //pool storage for commands
    VkCommandPool commandPool;
    //storage for allocation of command buffers
    std::vector<VkCommandBuffer> commandBuffers;

    //semaphores objects storage
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    //fences object storage
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    //keep track of the current rame
    size_t currentFrame = 0;



    //uniform buffers storage
    std::vector<VkBuffer> uniformBuffers;
    //memory for the uniform buffers
    std::vector<VkDeviceMemory> uniformBuffersMemory;








    //vertex buffer
    VkBuffer vertexBuffer;
    //vertex buffer memory
    VkDeviceMemory vertexBufferMemory;

    //index buffer
    VkBuffer indexBuffer;
    //index buffer memory
    VkDeviceMemory indexBufferMemory;




    //descriptor pool handle
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;


    glm::mat4 modelMatrix;
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;


    //variable use to check if window resize has happened
    bool framebufferResized = false;

    void initWindow() {

        //initialize the GLFW library
        glfwInit();

        //tell GLFW to not create OpenGL context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        //initialize the window
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        //set up callback to detect window resize
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        glfwSetKeyCallback(window, keyPressed);
    }

    //callback function which sets the value of framebufferResized variable thus notifies of a change in window size
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {

        //initialize the Vulkan library by creating an instance - the connection between application and the Vulkan library
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createTiles();
       

        setGame();
      
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        for (int i = 0; i < allMeshes.size(); i++) {

            createUniformModelBuffers(allMeshes[i]);
            createDescriptorSets(allMeshes[i]);
        }
        createCommandBuffers();
        createSyncObjects();

    }

    void mainLoop() {
        //keep the app running until an error occurs or the window is closed
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }
   
    static glm::vec2 normalize(glm::vec2 toNorm) {
        int mag = sqrt(toNorm[0] * toNorm[0] + toNorm[1] * toNorm[1]);
        return glm::vec2(toNorm[0]/mag,toNorm[1]/mag);


    }
  static  void loadModel(Mesh *mesh) {
        


        for (int i = 0; i < mesh->indices.size(); i++) {
            indices.push_back(mesh->indices[i]+vertices.size());
        }

        for (int i = 0; i < mesh->vertices.size();i++) {
            vertices.push_back(mesh->vertices[i]);
        }
       

        if (mesh->name == "Platform") {
            
            mesh->width = 100;
            mesh->height = 20;
            mesh->ubo.model = glm::mat4(1);
            mesh->ubo.model = glm::translate(mesh->ubo.model, glm::vec3(-((100 * mesh->scale[0]) / 2), -HEIGHT / 2, 0.0f));//add scale for height FIXHERE
            mesh->ubo.model = glm::scale(mesh->ubo.model, mesh->scale);
           



        }
        if (mesh->name == "Ball") {


            mesh->width = 20;
            mesh->height = 20;
            mesh->ubo.model = glm::mat4(1);
            mesh->ubo.model = glm::translate(mesh->ubo.model, glm::vec3(0, (-HEIGHT / 2)+20*mesh->scale[1], 0.0f));//add scale for height FIXHERE
            mesh->ubo.model = glm::scale(mesh->ubo.model, mesh->scale);
            mesh->direction = glm::vec2((rand() % 10 + 0)-5, 1.0f);
            mesh->direction = normalize(mesh->direction);

        }
      

    }




    //window surface can be changed so the swap chain will no longer be compatible with it, swap chain will have to be recreated but old versions of resources used by old swap chain will have to be deleted
    void cleanupSwapChain() {
        //destroy framebuffers
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        //free command buffers
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

        //destroy pipeline
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        //destroy pipeline layout
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        //destroy render pass
        vkDestroyRenderPass(device, renderPass, nullptr);

        //destroy image views
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        //destroy swapchain
        vkDestroySwapchainKHR(device, swapChain, nullptr);

        for (size_t i = 0; i < swapChainImages.size(); i++) {

            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);


        }

//Here



        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    }

    //clean up all resources
    void cleanup() {
        cleanupSwapChain();

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout[0], nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout[1], nullptr);
       
            vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device,vertexBufferMemory, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
    
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    //function which recreates swap chain after it becomes incompatible with window
    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createUniformBuffers();
        
        
        createDescriptorPool();
        createDescriptorSets();
        for (int i = 0; i < allMeshes.size(); i++) {

            createUniformModelBuffers(allMeshes[i]);
            createDescriptorSets(allMeshes[i]);
        }
        createCommandBuffers();
    }

    void createDescriptorSetLayout() {



        VkDescriptorSetLayoutBinding uboLayoutBinding[2]{};

        uboLayoutBinding[0].binding = 0;
        uboLayoutBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding[0].descriptorCount = 1;
        uboLayoutBinding[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;



        uboLayoutBinding[1].binding = 0;
        uboLayoutBinding[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding[1].descriptorCount = 1;
        uboLayoutBinding[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;


        VkDescriptorSetLayoutCreateInfo layoutInfo1{};
        layoutInfo1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo1.bindingCount = 1;
        layoutInfo1.pBindings = &uboLayoutBinding[0];
        
        VkDescriptorSetLayoutCreateInfo layoutInfo2{};
        layoutInfo2.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo2.bindingCount = 1;
        layoutInfo2.pBindings = &uboLayoutBinding[1];

        if (vkCreateDescriptorSetLayout(device, &layoutInfo1, nullptr, &descriptorSetLayout[0]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout");
        } 
        
        if (vkCreateDescriptorSetLayout(device, &layoutInfo2, nullptr, &descriptorSetLayout[1]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout");
        }

    }

    //create a buffer with given properties
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        //size of a buffer
        bufferInfo.size = size;
        //for which purposes data in a buffer will be used
        bufferInfo.usage = usage;
        //buffer will be used by one graphics family so set sharing mode to exclusive
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        //assign memory to the buffer
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);


    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {

            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ,
               uniformBuffers[i],uniformBuffersMemory[i]);

        }
    }



    void createUniformModelBuffers(Mesh *mesh) {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        mesh->uniformModelBuffers.resize(swapChainImages.size());
        mesh->uniformModelBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {

            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                mesh->uniformModelBuffers[i], mesh->uniformModelBuffersMemory[i]);

        }
    }


    void createVertexBuffer() {

        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;

        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);


        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

    }
    
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.size = size;

        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void createIndexBuffer() {

        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);


        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }


    void createInstance() {

        //check if validation layers were requested and are available
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        //optional structure with informations useful for the driver to optimize the application
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        //not optional structure which tells the Vulkan driver which global extensions and validation layers to use
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        //get required extensions and add them to a info structure
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;

        //if validation layers are enabled, modify VkInstanceCreateInfo struct instantiation to include the validation layer names 
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            //separate debug utils messenger for vkCreateInstance and vkDestroyInstance functions
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        //throw an error if failed to create an instance
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }
    /*
    since create/destroy debug utils messanger has to be called when valid instance exists,
    functions for creating/destroying instance cannot be debugged so
    this function('populateDebugMessengerCreateInfo') with messenger create info extracted
    will be used to setup separate debug messenger for creating/destroying instance and for CreateDebugUtilsMessengerEXT
    */
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        //specify which types of severities notify the callback function
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        //specify which types of messages notify callback
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        //pointer to the user's callback function
        createInfo.pfnUserCallback = debugCallback;
    }

    //function which tells Vulkan about the user's callback function
    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    //function which creates a surface for rendering
    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    //function which chooses suitable GPU
    void pickPhysicalDevice() {
        //check how many devices are available
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        //check which device is suitable
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    //create logical device, describe features to be used and specify which queues to create
    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        //create set of all unique queue families 
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        //priority value used to influence the scheduling of command buffer execution
        float queuePriority = 1.0f;
        //for each unique queue family create info and add it to a shared queue made from both families
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        //define and leave because no special features are needed for this program
        VkPhysicalDeviceFeatures deviceFeatures = {};

        //device create info structure
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        //add pointers to queue creation info
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        //add pointer to device features
        createInfo.pEnabledFeatures = &deviceFeatures;

        //enable device extensions
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        //implemented for older versions of Vulkan since the actual version does not make a distinction between instance and device specific validation layers
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        //instantiate the logical device 
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        //retrieve queue handle to interface with queues and store it in a class member
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    //function which creates swap chain which deals with presenting images on the screen
    void createSwapChain() {
        //check support for the swap chain
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        //set the options for swap chain
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        //specify number of images in the swap chain (choose minimum number, but to avoid waiting for internal operations to finish, acquire another image to render to)
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        //if imageCount exceeds maxImageCount, set imageCount to max possible
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        //fill in createInfo structure for swap chain
        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        //specifies the amount of layer the image consists of, always 1 unless developing stereoscopic 3D application
        createInfo.imageArrayLayers = 1;
        //render directly to images which means they are used as a color attachment
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        //specify how swap chain images are handled across multiple queue families
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        /*
        if the graphics queue family is different than presentation queue family use concurrent sharing mode which allows images to be used across multiple families
        without explicit ownership transfers which may present worse in the performance
        */
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        //exclusive sharing mode - image is owned by one queue family at a time
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        //specify informations for concurrent sharing mode

        //do not apply any transformation
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        //ignore the alpha channel
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        //ignore the colour of pixels that are obscured
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        //get final number of swap chain images
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        //resize handles storage and retrieve handles
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        //store swap chain image format and extent in member variables for future usage
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    //function which describes how to access the image and which part of the image to access
    void createImageViews() {
        //resize image views storage to fit all of the image views to be created
        swapChainImageViews.resize(swapChainImages.size());

        //create info for each swap chain image 
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            //treat images as 2D textures
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            //default mapping
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            //use image as colour targets, no mipmapping levels, no multiple layers
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            //create image view
            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    //function which specifies how many colours and depth buffers will be used, how many samples to use for them and how to handle their contents throughout the rendering operations.
    void createRenderPass() {
        //
        VkAttachmentDescription colorAttachment = {};
        //match the format with swap chain images format
        colorAttachment.format = swapChainImageFormat;
        //no multisampling so set to 1 sample
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        //before rendering clear the framebuffer
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        //store rendered contents in memory to be read later
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        //set stencil data to dont care because stencil buffer is not used in this application
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        //set initial image layout to undefined
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        //set final image layout to be presented in the swap chain
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        //attachment structure referenced to by subpass
        VkAttachmentReference colorAttachmentRef = {};
        //specify which attachment to reference - there is only one attachment in attachment description so set index to 0
        colorAttachmentRef.attachment = 0;
        //set the attachment to be used as a colour buffer when it is used by a subpass
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        //subpass structure
        VkSubpassDescription subpass = {};
        //declare that the subpass is a graphics subpass, not a compute subpass
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        //specify a reference to the colour attachment
        subpass.pColorAttachments = &colorAttachmentRef;

        //structure with informations about subpass dependencies
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        //wait for the swap chain to finish reading from the image before accessing it - wait on the colour attachment output stage
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        //specify the operations that should wait on the colour attachment output stage
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        //structure with informations about render pass
        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        //number of attachments
        renderPassInfo.attachmentCount = 1;
        //pointer to the colour attachment
        renderPassInfo.pAttachments = &colorAttachment;
        //number of subpasses
        renderPassInfo.subpassCount = 1;
        //pointer to a subpass
        renderPassInfo.pSubpasses = &subpass;
        //specify an array of dependencies
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    //function for creating graphics pipeline
    void createGraphicsPipeline() {
        //read in vertex shader
        auto vertShaderCode = readFile("shaders/vert.spv");
        //read in fragment shader
        auto fragShaderCode = readFile("shaders/frag.spv");

        //create shader modules for vertex shader and fragment shader
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        //to use shaders they have to be assigned to a specific pipeline stage
        //
        //create pipeline shader stage info for vertex shader
        VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        //assign vertex shader to vertex shader stage
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        //assign vertex shader module containing the code
        vertShaderStageInfo.module = vertShaderModule;
        //declare the entry point which is the function to be invoked
        vertShaderStageInfo.pName = "main";

        //create pipeline shader stage info for fragment shader
        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        //assign fragment shader to fragment shader stage
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        //assign fragment shader module containing the code
        fragShaderStageInfo.module = fragShaderModule;
        //declare the entry point which is the function to be invoked
        fragShaderStageInfo.pName = "main";

        //pack these 2 above structures to an array for future usage
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescription = Vertex::getAttributeDescriptions();

        //informations about the format of the vertex data which will be passed to the vertex shader
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescription.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescription.data();

        //specify what kind of geometry(point, line, triangle) will be drawn and if primitive restart(reusing vertices for optimization) should be enabled
        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        //transformation from the image to the framebuffer - describe the region of the framebuffer where rendering will be performed
        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        //specify the range of depth values to use for the framebuffer
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        //define region for storing pixels, any pixels outside the region will be discarded by the rasterizer
        VkRect2D scissor = {};
        //cover the whole framebuffer
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        //combine viewport and scissor rectangle into viewportState
        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        //structure with informations about rasterizer
        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        //discard fragments that are beyond the near and far planes instead of clamping them on these planes
        rasterizer.depthClampEnable = VK_FALSE;
        //if this is true the geometry never passes through the rasterizer stage, so set it to false
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        //specify how to generate fragments for geometry - fill the area of the polygon with fragments
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        //thickness of lines in terms of number of fragments
        rasterizer.lineWidth = 1.0f;
        //determine the type of face culling
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        //specify the vertex order for faces to be considered front-facing
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        //turn off altering the depth values
        rasterizer.depthBiasEnable = VK_FALSE;

        //disable multisampling 
        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        //structure specifying a pipeline colour blend attachment state
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        //specify which of the R,G,B,A components are enabled for writing
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        //disable blending so the source fragment's colour for that attachment is passed through unmodified
        colorBlendAttachment.blendEnable = VK_FALSE;

        //structure which specifies informations about colour blend state
        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        //mix the old and new value to produce a color instead of using bitwise operation
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        //specify the number of elements in pAttachments
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        //constant values which can be used in blending
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        //information about pipeline layout - uniform values which alter the behavior of shaders by passing the transformation matrix to the vertex shader or creating texture samplers in the fragment shader
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 2;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout[0];
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        //create pipeline layout
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        //structure with informations about pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        //number of stages in shaderStages array
        pipelineInfo.stageCount = 2;
        //pointer to a structure describing the set of shader stages to be included in the pipeline
        pipelineInfo.pStages = shaderStages;
        //pointers to structures describing parts of the pipeline
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        //set pipeline layout
        pipelineInfo.layout = pipelineLayout;
        //set render pass
        pipelineInfo.renderPass = renderPass;
        //index of a subpass in the render pass
        pipelineInfo.subpass = 0;
        //pipeline can be derived from existing pipeline but since there is only one pipeline, pass a null handle
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        //create graphics pipeline
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        //fragShaderModule and vertShaderModule can be destroyed after graphics pipeline has been created
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    //function which creates frame buffers for all of the images in the swap chain
    void createFramebuffers() {
        //resize framebuffers storage to fit all of the framebuffers
        swapChainFramebuffers.resize(swapChainImageViews.size());

        //iterate through image views and create framebuffers
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            //structur with informations about framebuffer
            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            //specify which render pass the framebuffer has to be compatible with
            framebufferInfo.renderPass = renderPass;
            //set the number of attachments
            framebufferInfo.attachmentCount = 1;
            //pointer to an array of handles 
            framebufferInfo.pAttachments = attachments;
            //define dimensions of framebuffer
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            //create framebuffer
            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    //function which creates command pool which is used to manage memory for storing buffers
    void createCommandPool() {
        //find queue families and use graphics queue family to record commands for drawing
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        //structure with informations about pool
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        //create command pool
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    //function which allocates and records the commands for each swap chain image
    void createCommandBuffers() {
        //resize command buffers storage to fit all of the command buffers
        commandBuffers.resize(swapChainFramebuffers.size());

        //function with informations required to allocate command buffers
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        //specify the command pool
        allocInfo.commandPool = commandPool;
        //specify if the allocated command buffers are primary or secondary command buffers, since in this program command buffers will not be called from other command buffers set command buffer level to primary
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        //specify number of buffers to allocate
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        //allocate command buffers
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        //record command buffers 
      
            //structure with details about the usage of command buffer
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            //begin recording command buffer
            for (size_t i = 0; i < commandBuffers.size(); i++) {

                if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                    throw std::runtime_error("failed to begin recording command buffer!");
                }

            //structure with informations about configuring the render pass
            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[i% swapChainFramebuffers.size()];
            //define the size of the render area which should match the size of the attachments
            renderPassInfo.renderArea.offset = { 0, 0 };
            renderPassInfo.renderArea.extent = swapChainExtent;

            //specify colour to use for VK_ATTACHMENT_LOAD_OP_CLEAR which is used as load operation for the colour attachment
            VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;
         
            //begin render pass, use VK_SUBPASS_CONTENTS_INLINE since secondary command buffers will not be used
            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            //bind the pipeline and scpecify that it is a graphics pipeline
            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
                VkBuffer vertexBuffers[] = { vertexBuffer };
                VkDeviceSize offsets[] = { 0 };
                  vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
                  vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);

                  int firstIndex = 0;
                  //int j = 1;
                  for (int j = 0; j < allMeshes.size(); j++) {





                      std::vector<VkDescriptorSet> sets;
                      sets.push_back(descriptorSets[i]);
                     
                      sets.push_back(allMeshes[j]->descriptorSets[i]);



                      vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, &sets[0], 0, nullptr);

                      // vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &allMeshes[j]->descriptorSets[i], 0, nullptr);

                       //draw a triangle, pass a command buffer, number of vertices, instanceCount used for instanced rendering (1 if not used), first vertex and first instance 
                      // vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(allMeshes[i-2]->vertices.size()), 1, 0, 0);
                      if (allMeshes[j]->name == "Tile")
                      {
                          Tile& tile = static_cast<Tile&>(*allMeshes[j]);

                          if (!tile.isVisible) {


                              continue;
                          }

                      }
                      vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(allMeshes[j]->indices.size()), 1, firstIndex, 0, 0);
                      firstIndex = firstIndex + allMeshes[j]->indices.size();
                  
                  }

            

            //end render pass
            vkCmdEndRenderPass(commandBuffers[i]);

            //end command buffer
            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void createSyncObjects() {
        //resize storages to fit all concurrent frames
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

        //structure with informations about semaphore
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        //structure with informations about fence
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        //create semaphores and fence
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void drawFrame() {
        //wait for the frame to be finished
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        /*
        acquire an image from the swap chain, pass logical device, swap chain from which an image will be acquired,
        maximum value of a 64 bit unsigned integer to disable the timeout for an image to become available and
        two parameters specifying synchronization objects which should be signaled when the presentation engine finished using image.
        the last parameter is a variable which holds the index of swap chain image which has become available
        */
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (reset)
            resetGame();
       

        //check if swap chain is compatible with the window, if not - recrete the swap chain
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        //check if previous frame is using given image
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        //mark the image as now being used by the current frame
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];


        animation();
        updateUniformBuffer(imageIndex);

      

        for (int i = 0; i < allMeshes.size(); i++) {
        updateUniformModelBuffer(imageIndex, allMeshes[i]);

    }
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        //specify which semaphores to wait for
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        //specify in which stages of the pipeline to wait
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        //specify which command buffers to use
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        //specify which semaphores to signal once the command buffer has finished execution
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        //submit command buffer to the graphics queue and use fences for synchronization
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        //configure presentation info
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        //specify which semaphores to wait for before presentation
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        //specify swap chains to present images to and the image for each swap chain
        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        //submit the request to present an image to the swap chain
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        //check if swap chain is out of date or window has been resized
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        //go to the next frame
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    static void createTiles() {
    
        for (int i = 0; i < 17; i++) {
            for (int j = 0; j < 4; j++) {
                Tile* tile = new Tile({
                    { {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },
                    { {100.0f, 100.0f}, {0.0f, 0.0f, 1.0f} },
                    { {100.0f, 0.0f}, {1.0f, 1.0f, 1.0f} },
                    { {0.0f, 100.0f}, {1.0f, 0.0f, 1.0f} }
                    },
                    { 2, 1, 0 , 0 , 1 ,3 },
                    "Tile", glm::vec3(1.0f, 1.0f, 1.0f));
                

                tile->width = 100;
                tile->height = 100;
                tile->ubo.model = glm::mat4(1);
                tile->ubo.model = glm::translate(tile->ubo.model, glm::vec3(-WIDTH/2+25+(i*(tile->width+10)), HEIGHT/2-((j+1)*(tile->height+10)), 0.0f));//add scale for height FIXHERE
                tile->ubo.model = glm::scale(tile->ubo.model, tile->scale);


                allTiles.push_back(tile);
            }
        }
    }

   void  resetGame() {
      start = 0;
      allMeshes.clear();
      allMeshes.push_back(&platform);
      allMeshes.push_back(&ball);

          for (int i = 0; i < allTiles.size();i++) {
    
              allTiles[i]->isVisible = true;

              allMeshes.push_back(allTiles[i]);
          }
      

      for (int i = 0; i < allMeshes.size(); i++) {
          loadModel(allMeshes[i]);
      }

      createCommandBuffers();
      reset = 0;
    }
   void setGame() {
       start = 0;
       allMeshes.clear();
       allMeshes.push_back(&platform);
       allMeshes.push_back(&ball);

       for (int i = 0; i < allTiles.size(); i++) {

           allTiles[i]->isVisible = true;

           allMeshes.push_back(allTiles[i]);
       }


       for (int i = 0; i < allMeshes.size(); i++) {
           loadModel(allMeshes[i]);
       }
   
   }


    static void keyPressed(GLFWwindow* window, int key, int scancode, int action, int mods) {


        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));

   


        if (action == GLFW_PRESS && key == GLFW_KEY_LEFT) {
            Left = 1;

         //   allMeshes[1]->ubo.view= glm::translate(allMeshes[1]->ubo.view, glm::vec3(-1.0f, 0.0f, 0.0f));

        }

        if (action == GLFW_RELEASE && key == GLFW_KEY_LEFT) {
            Left = 0;

        }

        
        if (action == GLFW_PRESS && key == GLFW_KEY_RIGHT) {
            Right = 1;
            //  allMeshes[1]->ubo.view = glm::translate(allMeshes[1]->ubo.view, glm::vec3(1.0f, 0.0f, 0.0f));

        }

        if (action == GLFW_RELEASE && key == GLFW_KEY_RIGHT) {
            Right = 0;

        }  
        
        if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) {

            if (!start) {
                start = 1;
            }
            else {
                reset = 1;

            }
        }

     


        
    
    }
    void collisionDetectionBallPlatform(Ball *meshBall, Platform *meshPlatform) {

        if (meshBall->ubo.model[3][0]                                                   <   meshPlatform->ubo.model[3][0] + meshPlatform->width * meshPlatform->scale[0] &&
            meshBall->ubo.model[3][0] + meshBall->width * meshBall->scale[0]       >   meshPlatform->ubo.model[3][0] &&
            std::abs(meshBall->ubo.model[3][1]      - (meshPlatform->ubo.model[3][1] + meshPlatform->height * meshPlatform->scale[1]))<=EPSILON )
        {
            meshBall->direction[1] *= -1;
        }
    
    }
    void collisionDetectionBallTile(Ball *meshBall , Tile *meshTile) {

        if (!meshTile->isVisible)
            return ;
        //colision from the top 
        if ((meshBall->ubo.model[3][0]                                              <     meshTile->ubo.model[3][0] + meshTile->width * meshTile->scale[0] &&
            meshBall->ubo.model[3][0] + meshBall->width * meshBall->scale[0]        >     meshTile->ubo.model[3][0] &&
            std::abs(meshBall->ubo.model[3][1] - (meshTile->ubo.model[3][1] + meshTile->height * meshTile->scale[1]))<=EPSILON))
        {
            meshBall->direction[1] *= -1;
            meshTile->isVisible = false;
            createCommandBuffers();
        }

        //from the bottom
        if ((meshBall->ubo.model[3][0]                                               <     meshTile->ubo.model[3][0] + meshTile->width * meshTile->scale[0] &&
            meshBall->ubo.model[3][0] + meshBall->width * meshBall->scale[0]        >     meshTile->ubo.model[3][0] &&
            std::abs((meshBall->ubo.model[3][1] + meshBall->height * meshBall->scale[1]) - meshTile->ubo.model[3][1]) <= EPSILON)) 
        
        {
            meshBall->direction[1] *= -1;
            meshTile->isVisible = false;
            createCommandBuffers();
        
        }

        //colision from the either side 
        if ((meshBall->ubo.model[3][1]                                               <    meshTile->ubo.model[3][1] + meshTile->height * meshTile->scale[1] && //from rigth
            meshBall->ubo.model[3][1] + meshBall->height * meshBall->scale[1]        >    meshTile->ubo.model[3][1] &&
            std::abs(meshBall->ubo.model[3][0] -( meshTile->ubo.model[3][0] + meshTile->width * meshTile->scale[0]))<=EPSILON)
            ||
            (meshBall->ubo.model[3][1]                                               <    meshTile->ubo.model[3][1] + meshTile->height * meshTile->scale[1] &&//from left
                meshBall->ubo.model[3][1] + meshBall->height * meshBall->scale[1]    >    meshTile->ubo.model[3][1] &&
              std::abs(  (meshBall->ubo.model[3][0] + meshBall->width * meshBall->scale[0] )   -   meshTile->ubo.model[3][0])<=EPSILON)
            )
        {
            meshBall->direction[0] *= -1;
            meshTile->isVisible = false;
            createCommandBuffers();
        }


       

    
    }
    void collisionDetectionPlatformWalls(Platform *meshPlatform) {
        if (Left) {
            if (meshPlatform->ubo.model[3][0] > -WIDTH / 2) {
                meshPlatform->ubo.model = glm::translate(meshPlatform->ubo.model, glm::vec3(-5.0f, 0.0f, 0.0f));


            }
        }
        if (Right) {
            if (meshPlatform->ubo.model[3][0] < (WIDTH / 2) - (meshPlatform->scale[0] * 100)) {
                meshPlatform->ubo.model = glm::translate(meshPlatform->ubo.model, glm::vec3(5.0f, 0.0f, 0.0f));

            }

        }
    }

    void collisionDetectionBallWalls(Ball *meshBall) {
        if (std::abs(meshBall->ubo.model[3][1] - (HEIGHT / 2))<=EPSILON) {
            meshBall->direction[1] *= -1;
        }
        if (std::abs( meshBall->ubo.model[3][0]- (-WIDTH / 2))<=EPSILON) {
            meshBall->direction[0] *= -1;
        }
        if (std::abs(meshBall->ubo.model[3][0] - (WIDTH / 2))<=EPSILON) {
            meshBall->direction[0] *= -1;
        }
        if (meshBall->ubo.model[3][1]< -HEIGHT / 2) {
            resetGame();
        }
        
    }
    void ballMovment(Ball *meshBall, Platform *meshPlatform) {
        if (start) {

            collisionDetectionBallWalls(meshBall);
            collisionDetectionBallPlatform(meshBall, meshPlatform);

            for (int i = 0; i < allTiles.size(); i++) {
                 collisionDetectionBallTile(meshBall,allTiles[i]);
            }
            int speed = 5;

            meshBall->ubo.model = glm::translate(meshBall->ubo.model, glm::vec3(meshBall->direction[0] * speed, meshBall->direction[1] * speed, 0.0f));

        }
        else{
            if (Left) {
                if (meshBall->ubo.model[3][0] > (-WIDTH + meshPlatform->scale[0] * meshPlatform->width) / 2) {
                    
                    meshBall->ubo.model = glm::translate(meshBall->ubo.model, glm::vec3(-5.0f, 0.0f, 0.0f));
                }

            }
            if (Right) {
                if (meshBall->ubo.model[3][0] < (WIDTH / 2)) {
                  
                        meshBall->ubo.model = glm::translate(meshBall->ubo.model, glm::vec3(5.0f, 0.0f, 0.0f));
                }
            }

        
        }

    
    
    }
    void animation() {
        collisionDetectionPlatformWalls(&platform);
        ballMovment(&ball, &platform);
    }
    void updateUniformBuffer(uint32_t currentImage) {

        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();

        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
      

        if (viewMatrix[0].x == NULL) {


            viewMatrix = glm::lookAt(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

            projectionMatrix = glm::ortho((float)-WIDTH / 2, (float)WIDTH / 2, (float)-HEIGHT / 2, (float)HEIGHT / 2, -1000.0f, 1000.0f);

        }

        ubo.view = viewMatrix;
        ubo.proj = projectionMatrix;

        //flip the image so it is not upside down. glm was designed for openGL which had Y coordinate of the clip coordinates inverted
       ubo.proj[1][1] *= -1;


        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    } 
    
    void updateUniformModelBuffer(uint32_t currentImage, Mesh* mesh) {

       
        UniformBufferModelObject ubo{};    




        void* data;
        vkMapMemory(device, mesh->uniformModelBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &mesh->ubo, sizeof(ubo));
        vkUnmapMemory(device, mesh->uniformModelBuffersMemory[currentImage]);
    }

    void createDescriptorPool() {

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(swapChainImages.size()*(allMeshes.size()+1));

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size()* (allMeshes.size()+1));

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool");
        }

    }

    void createDescriptorSets() {

        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout[0]);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();


      descriptorSets.resize(swapChainImages.size());

        if (vkAllocateDescriptorSets(device, &allocInfo,descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets");
        }

        for (size_t i = 0; i < swapChainImages.size(); i++) {

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo;

            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

        }

    }


    void createDescriptorSets(Mesh* mesh) {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout[1]);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();


        mesh->descriptorSets.resize(swapChainImages.size());

        if (vkAllocateDescriptorSets(device, &allocInfo, mesh->descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets");
        }

        for (size_t i = 0; i < swapChainImages.size(); i++) {

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = mesh->uniformModelBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = mesh->descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo;

            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

        }

    }



    //function which creates shader module
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        //fill in info about shader module - specify the length of the buffer with bytecode and pointer to the buffer
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        //change bytes to uint32_t to match the pointer type
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        //create shader module
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        //return created shader module
        return shaderModule;
    }

    //if available, choose preferred format and colour space, else choose the first one available 
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    //if available, choose triple buffering presentation mode, else choose mode which synchronizes displaying images with monitor's refresh rate
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    //specify the resolution of swap chain images by checking the actual size of the window
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        //get basic surface capabilities
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        //get available surface formats
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        //get available presentation modes
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    //check if a given device is suitable for operations to be performed
    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        //check if device supports required extensions
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        //check if swap chain support is adequate 
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            //at least one image format and one presentation mode is needed for this program
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    //function for checking if a device supports required extensions
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        //initialize extensionCount
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        //an array for storing details of extensions
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        //pass a number of extensions and an array to a function which will retrieve details of extensions
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        //check if all required extensions are available
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    //check which queue families are supported by the device and which of them support needed commands
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        //get number of queue families
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        //get properties of queue family
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        //find which families support VK_QUEUE_GRAPHICS_BIT
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            //check if a queue family has the capability of presenting to a window surface
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    //a function which returns required extensions based on whether validation layers are enabled or not
    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    //check if all of the requested validation layers are available
    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        //initialize layerCount
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        //an array for storing the properties of available layers
        std::vector<VkLayerProperties> availableLayers(layerCount);
        //pass a number of available validation layers and an array to a function which will retrieve properties of layers
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        //check if all of the layers in validationLayers are in availableLayers list
        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    //function for reading binary data from the file
    static std::vector<char> readFile(const std::string& filename) {
        //start reading at the end of the file and read the position to determine the size of the file and allocate a buffer
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        //go back to the beggining of the file and read all of the bytes at once
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    /*
    Debug callback function with VKAPI_ATTR and VKAPI_CALL to ensure the function has the right signature for Vulkan to call it.
    Depending on returned value, decide if the Vulkan call that triggered the validation layer should be aborted.

    first parameter  : specifies severity of a message which can be
                       - diagnostic message
                       - informational message
                       - message about behavior which can be a bug in application (warning)
                       - message about behavior which may cause crashes
    second parameter : specifies the type of a message which can be
                       - message about some event unrelated to the specification or performance
                       - message about violation of specification or possible mistake
                       - message which suggests that Vulkan is not optimally used
    third parameter  : refers to a structure which contains the details of the message
    fourth parameter : pointer to user's info about callback
    */
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

