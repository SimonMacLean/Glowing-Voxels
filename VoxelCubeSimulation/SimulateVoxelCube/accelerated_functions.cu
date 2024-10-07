#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <Windows.h>

constexpr int voxel_grid_size = 256;
constexpr float D_A = .5f;
constexpr float D_B = .25f;
extern __device__ half hsqrt(half);
extern __device__ float rsqrtf(float);
__device__ inline float len(const float3 v)
{
	return hsqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
__device__ inline float operator !(const float3 v)
{
    return rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
__device__ inline float operator &(const float3 v1, const float3 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
__device__ inline float3 operator ^(const float3 v1, const float3 v2)
{
    return make_float3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
}
__device__ inline float3 operator /(float3 v1, const float3 v2)
{
    v1.x /= v2.x;
    v1.y /= v2.y;
    v1.z /= v2.z;
    return v1;
}
__device__ inline float3 operator /(float3 v, const float s)
{
    v.x /= s;
    v.y /= s;
    v.z /= s;
    return v;
}
__device__ inline float3 operator *(float3 v1, const float3 v2)
{
    v1.x *= v2.x;
    v1.y *= v2.y;
    v1.z *= v2.z;
    return v1;
}
__device__ inline float3 operator *(float3 v, const float s)
{
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
__device__ inline float3 operator *(const float s, float3 v)
{
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
__device__ inline float3 operator +(float3 v1, const float3 v2)
{
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
    return v1;
}
__device__ inline float3 operator +(float3 v, const float s)
{
    v.x += s;
    v.y += s;
    v.z += s;
    return v;
}
__device__ inline float3 operator -(float3 v1, const float3 v2)
{
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
    return v1;
}
__device__ inline float3 operator -(float3 v, const float s)
{
    v.x -= s;
    v.y -= s;
    v.z -= s;
    return v;
}
__device__ inline float3 operator %(float3 v1, const float3 v2)
{
    v1.x = fmodf(v1.x, v2.x);
    v1.y = fmodf(v1.y, v2.y);
    v1.z = fmodf(v1.z, v2.z);
    return v1;
}
__device__ inline float2 operator +(float2 a, const float2 b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}
__device__ inline float2 operator -(float2 a, const float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}
__device__ inline float2 operator *(const float s, float2 a)
{
    a.x *= s;
    a.y *= s;
    return a;
}
__device__ inline float2 operator *(float2 a, const float s)
{
    a.x *= s;
    a.y *= s;
    return a;
}
__device__ inline void operator *=(float2& a, const float s)
{
    a.x *= s;
    a.y *= s;
}
__device__ inline void operator +=(float2& a, const float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
__device__ inline void operator -=(float2& a, const float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
__device__ inline int index(const int x, const int y, const int width)
{
    return x + y * width;
}
__device__ inline float3 rotate_vec(const float3 vec, const float3 axis, const float cos, const float sin)
{
    const float d = (1 - cos) * (axis & vec);
    return make_float3(
        d * axis.x + vec.x * cos + sin * (axis.y * vec.z - axis.z * vec.y),
        d * axis.y + vec.y * cos + sin * (axis.z * vec.x - axis.x * vec.z),
        d * axis.z + vec.z * cos + sin * (axis.x * vec.y - axis.y * vec.x));
}
__global__ void get_direction_length_inv(float* ray_length_invs, const float focal_length, const int width, const int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    ray_length_invs[index(i, height - 1 - j, width)] = rsqrtf(
        focal_length * focal_length + ((j - height / 2.f) * (j - height / 2.f) + (i - width / 2.f) * (i - width / 2.f)) / height / height);
}
__global__ void get_direction(float3* directions, const float* ray_length_invs, const float3 x, const float3 y, const float3 z,
    const float focal_length, const int width, const int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    const int h = index(i, height - 1 - j, width);
    directions[h] = (z * focal_length + y * ((j - height / 2.f) / height) + x * ((i - width / 2.f) / height)) * ray_length_invs[h];
}

__global__ void simulate_ray(const float3* directions, const float3* voxelOpacities, const float3 camera_location, uchar3* pixels, const int width, const int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    const int h = index(i, j, width);
    const float3 direction = directions[h];
    const float3 camera = camera_location + voxel_grid_size/2;
    const float2 distances_of_x_intersection = direction.x == 0 ? make_float2(-FLT_MAX, FLT_MAX) : make_float2(camera.x / -direction.x, (voxel_grid_size - camera.x) / direction.x);
    const float2 distances_of_y_intersection = direction.y == 0 ? make_float2(-FLT_MAX, FLT_MAX) : make_float2(camera.y / -direction.y, (voxel_grid_size - camera.y) / direction.y);
    const float2 distances_of_z_intersection = direction.z == 0 ? make_float2(-FLT_MAX, FLT_MAX) : make_float2(camera.z / -direction.z, (voxel_grid_size - camera.z) / direction.z);
    const float distanceAtExitX = fmaxf(distances_of_x_intersection.x, distances_of_x_intersection.y);
    const float distanceAtExitY = fmaxf(distances_of_y_intersection.x, distances_of_y_intersection.y);
    const float distanceAtExitZ = fmaxf(distances_of_z_intersection.x, distances_of_z_intersection.y);
    int integer_index = distanceAtExitX < distanceAtExitY
	                       ? (distanceAtExitX < distanceAtExitZ ? 0 : 2)
	                       : distanceAtExitY < distanceAtExitZ
	                       ? 1
	                       : 2;
    const float distance_at_exit = fminf(fminf(distanceAtExitX, distanceAtExitY), distanceAtExitZ);
    float3 current_point = camera + direction * distance_at_exit;
    current_point = make_float3(integer_index== 0 ? roundf(current_point.x) : current_point.x, integer_index == 1 ? roundf(current_point.y) : current_point.y, integer_index == 2 ? roundf(current_point.z) : current_point.z);
    float3 ln_light_passing_through = make_float3(0, 0, 0);
    while (current_point.x >= 0 && current_point.x <= voxel_grid_size && current_point.y >= 0 && current_point.y <= voxel_grid_size && current_point.z >= 0 && current_point.z <= voxel_grid_size && (current_point.x - camera.x) / direction.x > 0)
    {
	    const float x_distance_to_next_plane = direction.x == 0 ? FLT_MAX : ((direction.x > 0 ? ceilf(current_point.x - 1) : floorf(current_point.x + 1)) - current_point.x) / -direction.x;
        const float y_distance_to_next_plane = direction.y == 0 ? FLT_MAX : ((direction.y > 0 ? ceilf(current_point.y - 1) : floorf(current_point.y + 1)) - current_point.y) / -direction.y;
        const float z_distance_to_next_plane = direction.z == 0 ? FLT_MAX : ((direction.z > 0 ? ceilf(current_point.z - 1) : floorf(current_point.z + 1)) - current_point.z) / -direction.z;
        float distance_to_next_plane = fminf(fminf(x_distance_to_next_plane, y_distance_to_next_plane), z_distance_to_next_plane);
        integer_index = x_distance_to_next_plane < y_distance_to_next_plane ? (x_distance_to_next_plane < z_distance_to_next_plane ? 0 : 2) : y_distance_to_next_plane < z_distance_to_next_plane ? 1 : 2;
        float3 next_point = current_point - direction * distance_to_next_plane;
        next_point = make_float3(integer_index == 0 ? roundf(next_point.x) : next_point.x, integer_index == 1 ? roundf(next_point.y) : next_point.y, integer_index == 2 ? roundf(next_point.z) : next_point.z);
        if(next_point.x < 0 || next_point.x > voxel_grid_size || next_point.y < 0 || next_point.y > voxel_grid_size || next_point.z < 0 || next_point.z > voxel_grid_size)
			break;
        if((next_point.x - camera.x) / direction.x < 0)
            distance_to_next_plane = len(camera - current_point);
        const float3 center_point = (current_point + next_point) / 2.0f;
	    int x_index = ceil(center_point.x - 1);
	    int y_index = ceil(center_point.y - 1);
	    int z_index = ceil(center_point.z - 1);
	    int index = (x_index * voxel_grid_size + y_index) * voxel_grid_size + z_index;
        const float3 voxel_opacity = voxelOpacities[index];
        ln_light_passing_through = ln_light_passing_through + voxel_opacity * distance_to_next_plane/ voxel_grid_size;
        current_point = next_point;
    }
    //Reverse the order of R G and B because the color is in BGR format
    pixels[h] = make_uchar3(static_cast<unsigned char>(255.f * (1-expf(ln_light_passing_through.z))),
                            static_cast<unsigned char>(255.f * (1-expf(ln_light_passing_through.y))),
                            static_cast<unsigned char>(255.f * (1-expf(ln_light_passing_through.x))));
}
__device__ inline int index_3d(const int i, const int j, const int k)
{
    return (i * voxel_grid_size + j) * voxel_grid_size + k;
}
__device__ inline float2 laplacian(const float2* u, int i, int j, int k)
{
    
	constexpr float face_neighbor_weight = 1.f / 26;
	constexpr float edge_neighbor_weight = 1.f / 26;
	constexpr float corner_neighbor_weight = 1.f / 26;
    float2 sum = make_float2(0, 0);
	const int i_prev = i - (i > 0);
	const int j_prev = j - (j > 0);
	const int k_prev = k - (k > 0);
	const int i_next = i + (i < voxel_grid_size - 1);
	const int j_next = j + (j < voxel_grid_size - 1);
	const int k_next = k + (k < voxel_grid_size - 1);
    sum += u[index_3d(i_prev, j_prev, k_prev)] * corner_neighbor_weight;
    sum += u[index_3d(i_prev, j_prev, k)] * edge_neighbor_weight;
    sum += u[index_3d(i_prev, j_prev, k_next)] * corner_neighbor_weight;
    sum += u[index_3d(i_prev, j, k_prev)] * edge_neighbor_weight;
    sum += u[index_3d(i_prev, j, k)] * face_neighbor_weight;
    sum += u[index_3d(i_prev, j, k_next)] * edge_neighbor_weight;
    sum += u[index_3d(i_prev, j_next, k_prev)] * corner_neighbor_weight;
    sum += u[index_3d(i_prev, j_next, k)] * edge_neighbor_weight;
    sum += u[index_3d(i_prev, j_next, k_next)] * corner_neighbor_weight;
    sum += u[index_3d(i, j_prev, k_prev)] * edge_neighbor_weight;
    sum += u[index_3d(i, j_prev, k)] * face_neighbor_weight;
    sum += u[index_3d(i, j_prev, k_next)] * edge_neighbor_weight;
    sum += u[index_3d(i, j, k_prev)] * face_neighbor_weight;
    sum -= u[index_3d(i, j, k)];
    sum += u[index_3d(i, j, k_next)] * face_neighbor_weight;
    sum += u[index_3d(i, j_next, k_prev)] * edge_neighbor_weight;
    sum += u[index_3d(i, j_next, k)] * face_neighbor_weight;
    sum += u[index_3d(i, j_next, k_next)] * edge_neighbor_weight;
    sum += u[index_3d(i_next, j_prev, k_prev)] * corner_neighbor_weight;
    sum += u[index_3d(i_next, j_prev, k)] * edge_neighbor_weight;
    sum += u[index_3d(i_next, j_prev, k_next)] * corner_neighbor_weight;
    sum += u[index_3d(i_next, j, k_prev)] * edge_neighbor_weight;
    sum += u[index_3d(i_next, j, k)] * face_neighbor_weight;
    sum += u[index_3d(i_next, j, k_next)] * edge_neighbor_weight;
    sum += u[index_3d(i_next, j_next, k_prev)] * corner_neighbor_weight;
    sum += u[index_3d(i_next, j_next, k)] * edge_neighbor_weight;
    sum += u[index_3d(i_next, j_next, k_next)] * corner_neighbor_weight;
    /*
    float2 sum = make_float2(0, 0);
    const int j_prev = j - (j > 0);
    const int k_prev = k - (k > 0);
    const int j_next = j + (j < voxel_grid_size - 1);
    const int k_next = k + (k < voxel_grid_size - 1);
    constexpr float face_neighbor_weight = 0.2f;
    constexpr float edge_neighbor_weight = 0.05f;
    sum += u[index_3d(i, j_prev, k_prev)] * edge_neighbor_weight;
    sum += u[index_3d(i, j_prev, k)] * face_neighbor_weight;
    sum += u[index_3d(i, j_prev, k_next)] * edge_neighbor_weight;
    sum += u[index_3d(i, j, k_prev)] * face_neighbor_weight;
    sum -= u[index_3d(i, j, k)];
    sum += u[index_3d(i, j, k_next)] * face_neighbor_weight;
    sum += u[index_3d(i, j_next, k_prev)] * edge_neighbor_weight;
    sum += u[index_3d(i, j_next, k)] * face_neighbor_weight;
    sum += u[index_3d(i, j_next, k_next)] * edge_neighbor_weight;
    */
    return sum;
}
__device__ inline float clamp(float x, float minn, float maxx)
{
    return min(maxx, max(minn, x));
}
__global__ void react_diffuse(const float2* u_in, float2* u_out, const float dt, const float feed_rate, const float kill_rate)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= voxel_grid_size || j >= voxel_grid_size || k >= voxel_grid_size) return;
	const int idx = index_3d(i, j, k);
	const float A_amount = u_in[idx].x;
	const float B_amount = u_in[idx].y;
	const float2 nabla = laplacian(u_in, i, j, k);
	const float new_A_amount = A_amount + (D_A * nabla.x - A_amount * B_amount * B_amount + feed_rate * (1 - A_amount)) * dt;
	const float new_B_amount = B_amount + (D_B * nabla.y + A_amount * B_amount * B_amount - (kill_rate + feed_rate) * B_amount) * dt;
    u_out[idx] = make_float2(clamp(new_A_amount, 0, 1), clamp(new_B_amount, 0, 1));
}
__device__ inline float3 log(const float3 v)
{
	return make_float3(logf(v.x), logf(v.y), logf(v.z));
}
__global__ void copy_to_voxel(const float2* u, float3* voxels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = index_3d(i, j, k);
    voxels[idx] = log(float3{ 1,1,1 } - 0.99f*make_float3(u[idx].y, 0.f, 0.f));
}