#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {

    constexpr float camera_fov = 70.0f;
    constexpr float camera_near_plane = 0.01f;
    constexpr float camera_far_plane = 100.0f;

    struct Matrix {
        float m[4][4];
    };

    struct Vector {
        float x, y, z;
    };

    struct Vertex {
        Vector position;
        Vector color;
        // NOTE: You can add more attributes
    };

// NOTE: These variable will be available to shaders through push constant uniform
    struct ShaderConstants {
        Matrix projection;
        Matrix transform;
        Vector color;
    };

    struct VulkanBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    };

    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

// NOTE: Declare buffers and other variables here
    VulkanBuffer vertex_buffer;
    VulkanBuffer index_buffer;

    float child_scale_val = 0.4f;
    float orbit_radius = 1.5f;
    float child_scale_val_1 = 0.6f;
    float orbit_radius_1 = 2.5f;
    float rotation_speed = 1.0f;
    float total_elapsed_time = 0.0f;

    Matrix identity() {
        Matrix result{};

        result.m[0][0] = 1.0f;
        result.m[1][1] = 1.0f;
        result.m[2][2] = 1.0f;
        result.m[3][3] = 1.0f;

        return result;
    }

    Matrix projection(float fov, float aspect_ratio, float near, float far) {
        Matrix result{};

        const float radians = fov * M_PI / 180.0f;
        const float cot = 1.0f / tanf(radians / 2.0f);

        result.m[0][0] = cot / aspect_ratio;
        result.m[1][1] = cot;
        result.m[2][3] = 1.0f;

        result.m[2][2] = far / (far - near);
        result.m[3][2] = (-near * far) / (far - near);

        return result;
    }

    Matrix translation(Vector vector) {
        Matrix result = identity();

        result.m[3][0] = vector.x;
        result.m[3][1] = vector.y;
        result.m[3][2] = vector.z;

        return result;
    }

    Matrix rotation(Vector axis, float angle) {
        Matrix result{};

        float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

        axis.x /= length;
        axis.y /= length;
        axis.z /= length;

        float sina = sinf(angle);
        float cosa = cosf(angle);
        float cosv = 1.0f - cosa;

        result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
        result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
        result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

        result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
        result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
        result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

        result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
        result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
        result.m[2][2] = (axis.z * axis.z * cosv) + cosa;

        result.m[3][3] = 1.0f;

        return result;
    }

    // функция для создания матрицы масштабирования
    Matrix scale(Vector s) {
        Matrix result = identity();
        result.m[0][0] = s.x;
        result.m[1][1] = s.y;
        result.m[2][2] = s.z;
        return result;
    }

    Matrix multiply(const Matrix& a, const Matrix& b) {
        Matrix result{};

        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                for (int k = 0; k < 4; k++) {
                    result.m[j][i] += a.m[j][k] * b.m[k][i];
                }
            }
        }

        return result;
    }

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
    VkShaderModule loadShaderModule(const char* path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{//структура для создания шейдерного модуля
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = size,
                .pCode = buffer.data(),
        };

        VkShaderModule result;
        if (vkCreateShaderModule(veekay::app.vk_device, &//создает на GPU объект шейдерного модуля
                info, nullptr, &result) != VK_SUCCESS) {
            return nullptr;
        }

        return result;
    }

    VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage) {
        VkDevice& device = veekay::app.vk_device;
        VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        VulkanBuffer result{};

        {
            // NOTE: We create a buffer of specific usage with specified size
            VkBufferCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = size,
                    .usage = usage,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan buffer\n";
                return {};
            }
        }

        // NOTE: Creating a buffer does not allocate memory,
        //       only a buffer **object** was created.
        //       So, we allocate memory for the buffer

        {
            // NOTE: Ask buffer about its memory requirements
            VkMemoryRequirements requirements;
            vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

            // NOTE: Ask GPU about types of memory it supports
            VkPhysicalDeviceMemoryProperties properties;
            vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

            // NOTE: We want type of memory which is visible to both CPU and GPU
            // NOTE: HOST is CPU, DEVICE is GPU; we are interested in "CPU" visible memory
            // NOTE: COHERENT means that CPU cache will be invalidated upon mapping memory region
            const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            // NOTE: Linear search through types of memory until
            //       one type matches the requirements, thats the index of memory type
            uint32_t index = UINT_MAX;
            for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
                const VkMemoryType& type = properties.memoryTypes[i];
                //Проверяем, поддерживает ли i-тый тип памяти требования нашего буфера и соответствует ли он нашим флагам (виден для CPU)
                if ((requirements.memoryTypeBits & (1 << i)) &&
                    (type.propertyFlags & flags) == flags) {
                    index = i;
                    break;
                }
            }

            if (index == UINT_MAX) {
                std::cerr << "Failed to find required memory type to allocate Vulkan buffer\n";
                return {};
            }

            // NOTE: Allocate required memory amount in appropriate memory type
            VkMemoryAllocateInfo info{//заполняем структуру на выделение памяти
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .allocationSize = requirements.size,
                    .memoryTypeIndex = index,
            };

            // выделяем память на GPU
            if (vkAllocateMemory(device, &info, nullptr, &result.memory) != VK_SUCCESS) {
                std::cerr << "Failed to allocate Vulkan buffer memory\n";
                return {};
            }

            //связываем буфер
            // NOTE: Link allocated memory with a buffer
            if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) {
                std::cerr << "Failed to bind Vulkan  buffer memory\n";
                return {};
            }

            // NOTE: Get pointer to allocated memory
            void* device_data;
            vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);
            //копируем данные из оперативной памяти в память GPU
            memcpy(device_data, data, size);

            vkUnmapMemory(device, result.memory);
        }

        return result;
    }

    void destroyBuffer(const VulkanBuffer& buffer) {
        VkDevice& device = veekay::app.vk_device;

        vkFreeMemory(device, buffer.memory, nullptr);
        vkDestroyBuffer(device, buffer.buffer, nullptr);
    }

    void initialize() {
        VkDevice& device = veekay::app.vk_device;
        VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        { // NOTE: Build graphics pipeline
            vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
            if (!vertex_shader_module) {
                std::cerr << "Failed to load Vulkan vertex shader from file\n";
                veekay::app.running = false;
                return;
            }

            fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
            if (!fragment_shader_module) {
                std::cerr << "Failed to load Vulkan fragment shader from file\n";
                veekay::app.running = false;
                return;
            }

            VkPipelineShaderStageCreateInfo stage_infos[2];

            // NOTE: Vertex shader stage
            stage_infos[0] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertex_shader_module,
                    .pName = "main",
            };

            // NOTE: Fragment shader stage
            stage_infos[1] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragment_shader_module,
                    .pName = "main",
            };

            // NOTE: How many bytes does a vertex take?
            VkVertexInputBindingDescription buffer_binding{
                    .binding = 0,
                    .stride = sizeof(Vertex),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            // NOTE: Declare vertex attributes
            VkVertexInputAttributeDescription attributes[] = {
                    {
                            .location = 0, // NOTE: First attribute
                            .binding = 0, // NOTE: First vertex buffer
                            .format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
                            .offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
                    },
                    // NOTE: If you want more attributes per vertex,б т declare them here
#if 1
                    {
				.location = 1, // NOTE: Second attribute
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color),
			},
#endif
            };

            // NOTE: Bring
            VkPipelineVertexInputStateCreateInfo input_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    .vertexBindingDescriptionCount = 1,
                    .pVertexBindingDescriptions = &buffer_binding,
                    .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
                    .pVertexAttributeDescriptions = attributes,
            };

            // NOTE: Every three vertices make up a triangle,
            //       so our vertex buffer contains a "list of triangles"
            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,//Бери вершины по три штуки и рисуй из каждой тройки независимый треугольник
            };

            // NOTE: Declare clockwise triangle order as front-facing
            //       Discard triangles that are facing away
            //       Fill triangles, don't draw lines instaed
            VkPipelineRasterizationStateCreateInfo raster_info{//3D-треугольники превращаются в 2D-пиксели
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    .polygonMode = VK_POLYGON_MODE_FILL,//закрашивать треугольники
                    .cullMode = VK_CULL_MODE_BACK_BIT,//отсечение задних граней
                    .frontFace = VK_FRONT_FACE_CLOCKWISE,//лицевая сторона
                    .lineWidth = 1.0f,
            };

            // NOTE: Use 1 sample per pixel
            VkPipelineMultisampleStateCreateInfo sample_info{//мультисэмплинг
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                    .sampleShadingEnable = false,
                    .minSampleShading = 1.0f,
            };

            VkViewport viewport{
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(veekay::app.window_width),
                    .height = static_cast<float>(veekay::app.window_height),
                    .minDepth = 0.0f,
                    .maxDepth = 1.0f,
            };

            VkRect2D scissor{
                    .offset = {0, 0},
                    .extent = {veekay::app.window_width, veekay::app.window_height},
            };

            // NOTE: Let rasterizer draw on the entire window
            VkPipelineViewportStateCreateInfo viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

                    .viewportCount = 1,
                    .pViewports = &viewport,

                    .scissorCount = 1,
                    .pScissors = &scissor,
            };

            // NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
            VkPipelineDepthStencilStateCreateInfo depth_info{//тест глубины
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            // NOTE: Let fragment shader write all the color channels
            VkPipelineColorBlendAttachmentState attachment_info{
                    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT,
            };

            // NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
            VkPipelineColorBlendStateCreateInfo blend_info{//смешивание цветов
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

                    .logicOpEnable = false,
                    .logicOp = VK_LOGIC_OP_COPY,

                    .attachmentCount = 1,
                    .pAttachments = &attachment_info
            };

            // NOTE: Declare constant memory region visible to vertex and fragment shaders
            VkPushConstantRange push_constants{//определяет какие данные мы будет передавать в шейдер во врмя рендеринга
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                    .size = sizeof(ShaderConstants),
            };

            // NOTE: Declare external data sources, only push constants this time
            VkPipelineLayoutCreateInfo layout_info{//созданный объект по push_contants
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .pushConstantRangeCount = 1,
                    .pPushConstantRanges = &push_constants,
            };

            // NOTE: Create pipeline layout
            if (vkCreatePipelineLayout(device, &layout_info,
                                       nullptr, &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline layout\n";
                veekay::app.running = false;
                return;
            }

            VkGraphicsPipelineCreateInfo info{//собираем настройки все в одну настройку, структуру
                    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    .stageCount = 2,
                    .pStages = stage_infos,
                    .pVertexInputState = &input_state_info,
                    .pInputAssemblyState = &assembly_state_info,
                    .pViewportState = &viewport_info,
                    .pRasterizationState = &raster_info,
                    .pMultisampleState = &sample_info,
                    .pDepthStencilState = &depth_info,
                    .pColorBlendState = &blend_info,
                    .layout = pipeline_layout,
                    .renderPass = veekay::app.vk_render_pass,
            };

            // NOTE: Create graphics pipeline
            if (vkCreateGraphicsPipelines(device, nullptr,//объект со всеми настройками, который будем включать в функции render
                                          1, &info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        // TODO: You define model vertices and create buffers here
        // TODO: Index buffer has to be created here too
        // NOTE: Look for createBuffer function

        // (v0)------(v1)
        //  |  \       |
        //  |   `--,   |
        //  |       \  |
        // (v3)------(v2)
        const std::vector<Vertex> pyramid_vertices = {
                {{ -0.5f, 0.5f, -0.5f}, {1.0f, 0.3f,0.0f}},
                {{ 0.0f, -0.5f, 0.0f}, {1.0f, 0.0f,0.7f}},
                {{ 0.5f,  0.5f, -0.5f}, {.0f, 0.1f,0.4f}},
                {{0.5f,  0.5f, 0.5f}, {1.0f, 0.0f,0.9f}},
                {{ -0.5f,  0.5f, 0.5f}, {1.0f, 0.5f,0.0f}}
        };
        // Основание (2 треугольника = 6 индексов) + 4 боковые грани (4*3=12 индексов) = 18
        const std::vector<uint32_t> pyramid_indices = {
                0, 1, 2,
                2, 1, 3,
                3, 1, 4,
                4, 1, 0
        };
        vertex_buffer = createBuffer(pyramid_vertices.size() * sizeof(Vertex), (void*)pyramid_vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        index_buffer = createBuffer(pyramid_indices.size() * sizeof(uint32_t), (void*)pyramid_indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    }

    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        // NOTE: Destroy resources here, do not cause leaks in your program!
        destroyBuffer(index_buffer);
        destroyBuffer(vertex_buffer);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        ImGui::Begin("Controls:");
        ImGui::SliderFloat("Child Scale1", &child_scale_val, 0.1f, 1.0f);
        ImGui::SliderFloat("Orbit Radius1", &orbit_radius, 0.5f, 5.0f);
        ImGui::SliderFloat("Child Scale", &child_scale_val_1, 0.1f, 1.0f);
        ImGui::SliderFloat("Orbit Radius", &orbit_radius_1, 0.5f, 5.0f);
        ImGui::SliderFloat("Rotation Speed", &rotation_speed, 0.0f, 5.0f);
        ImGui::End();

        total_elapsed_time = (float)time;
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);//очищает
        VkCommandBufferBeginInfo begin_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,};//запись нового списка
        vkBeginCommandBuffer(cmd, &begin_info);

        VkClearValue clear_values[2];
        clear_values[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};//цвет фона
        clear_values[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo render_pass_info{ .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = veekay::app.vk_render_pass, .framebuffer = framebuffer, .renderArea = {.extent = {veekay::app.window_width, veekay::app.window_height},}, .clearValueCount = 2, .pClearValues = clear_values,};
        vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
            vkCmdBindIndexBuffer(cmd, index_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);

            ShaderConstants constants;
            constants.projection = projection(
                    camera_fov,
                    float(veekay::app.window_width) / float(veekay::app.window_height),
                    camera_near_plane, camera_far_plane);//матрица перспективы из 3d в 2d

            Matrix camera_rotation = rotation({1.0f, 0.0f, 0.0f}, 0.0f); // Вращение вокруг X-оси
            Matrix world_transform = multiply(camera_rotation, translation({0.0f, 0.0f, 5.0f}));

            float angle = total_elapsed_time * rotation_speed;
            Matrix system_rotation = rotation({0.0f, 0.0f, 1.0f}, angle);

            // первая пирамида
            Matrix scene_scale = scale({2.0f, 2.0f, 2.0f});
            Matrix parent_model = multiply(scene_scale,system_rotation);
            constants.transform = multiply(parent_model, world_transform);
            constants.color = {1.0f, 0.5f, 0.2f};
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShaderConstants), &constants);
            vkCmdDrawIndexed(cmd, 12, 1, 0, 0, 0);

            //первая дочерняя пирамида
            Matrix scale_m = scale({child_scale_val, child_scale_val, child_scale_val});
            Matrix orbit_m = translation({orbit_radius, 0.0f, 0.0f});
            Matrix child_local_transform = multiply(scale_m, orbit_m);
            Matrix child_model = multiply(child_local_transform, parent_model);
            constants.transform = multiply(child_model, world_transform);
            constants.color = {0.2f, 0.5f, 1.0f};
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShaderConstants), &constants);
            vkCmdDrawIndexed(cmd, 12, 1, 0, 0, 0);

            //первая дочерняя пирамида
            Matrix scale_m_1 = scale({child_scale_val_1, child_scale_val_1, child_scale_val_1});
            Matrix orbit_m_1 = translation({orbit_radius_1, 0.0f, 0.0f});
            Matrix child_local_transform_1 = multiply(scale_m_1, orbit_m_1);
            Matrix child_model_1 = multiply(child_local_transform_1, parent_model);
            constants.transform = multiply(child_model_1, world_transform);
            constants.color = {0.1f, 0.9f, 1.0f};
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShaderConstants), &constants);
            vkCmdDrawIndexed(cmd, 12, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace

int main() {
    return veekay::run({
                               .init = initialize,
                               .shutdown = shutdown,
                               .update = update,
                               .render = render,
                       });
}