#include "VoxelGridRenderer.h"
#include <corecrt_math.h>
#include <cstdio>
#include <msclr/marshal_cppstd.h>

#include "ReactionDiffusionSimulator.h"

cudaError_t get_direction_lengths(dim3 grid_size, dim3 block_size, float** dev_ray_length_invs,
                                  float focal_length, int draw_width, int draw_height)
{
	extern void get_direction_length_inv(float* ray_length_invs, const float focal_len, const int width, const int height);

	void* gdl_args[] = { dev_ray_length_invs, &focal_length, &draw_width, &draw_height };

	cudaError_t cuda_status = cudaLaunchKernel((const void*)get_direction_length_inv, grid_size, block_size, gdl_args);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "get_direction_length launch failed: %s\n", cudaGetErrorString(cuda_status));
		return cuda_status;
	}
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching get_direction_length!\n",
			cuda_status);
	return cuda_status;

}

cudaError_t get_directions(dim3 grid_size, dim3 block_size, float3** dev_ray_directions,
	float** dev_ray_length_invs, float3 camera_relative_x, float3 camera_relative_y,
	float3 camera_relative_z, float focal_length, int draw_width, int draw_height)
{
	extern void get_direction(float3 * directions, const float* ray_length_invs, const float3 x, const float3 y, const float3 z,
		const float focal_length, const int width, const int height);
	void* gd_args[] = {
		dev_ray_directions, dev_ray_length_invs, &camera_relative_x, &camera_relative_y, &camera_relative_z,
		&focal_length, &draw_width, &draw_height
	};
	cudaError_t cuda_status = cudaLaunchKernel((const void*)get_direction, grid_size, block_size, gd_args);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "get_direction launch failed: %s\n", cudaGetErrorString(cuda_status));
		return cuda_status;
	}
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching get_direction!\n", cuda_status);
	return cuda_status;
}

cudaError_t copy_pixels(unsigned char* dev_pixel_values, interior_ptr<cli::array<unsigned char>^> pixels,
	int draw_width, int draw_height)
{
	pin_ptr<unsigned char> pixels_start = &(*pixels)[0];
	cudaError_t cuda_status = cudaMemcpy(pixels_start, dev_pixel_values,
		draw_width * draw_height * sizeof(uchar3),
		cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");
	return cuda_status;
}

cudaError_t alloc_device_memory(float3** dev_ray_directions, float** dev_ray_lengths, uchar3** dev_pixels, float3** dev_voxel_grid,
	int draw_width, int draw_height, int voxel_grid_size)
{
	cudaError_t cuda_status = cudaMalloc(dev_ray_directions, draw_width * draw_height * sizeof(float3));
	if (cuda_status != cudaSuccess)
		return cuda_status;
	cuda_status = cudaMalloc(dev_ray_lengths, draw_width * draw_height * sizeof(float));
	if (cuda_status != cudaSuccess)
		return cuda_status;
	cuda_status = cudaMalloc(dev_pixels, draw_width * draw_height * sizeof(uchar3));
	if (cuda_status != cudaSuccess)
		return cuda_status;
	cuda_status = cudaMalloc(dev_voxel_grid, voxel_grid_size * voxel_grid_size * voxel_grid_size * sizeof(float3));
	return cuda_status;
}

cudaError_t alloc_2d_device_memory(float3** dev_ray_directions, float** dev_ray_lengths, uchar3** dev_pixels,
	int draw_width, int draw_height)
{
	cudaError_t cuda_status = cudaMalloc(dev_ray_directions, draw_width * draw_height * sizeof(float3));
	if (cuda_status != cudaSuccess)
		return cuda_status;
	cuda_status = cudaMalloc(dev_ray_lengths, draw_width * draw_height * sizeof(float));
	if (cuda_status != cudaSuccess)
		return cuda_status;
	cuda_status = cudaMalloc(dev_pixels, draw_width * draw_height * sizeof(uchar3));
	return cuda_status;
}

cudaError_t simulate_voxel_cube(dim3 grid_size, dim3 block_size, float3* dev_ray_directions, float3* dev_voxel_grid, uchar3* dev_pixels, float3 camera_location, int draw_width, int draw_height)
{
	extern void simulate_ray(const float3 * directions, const float3* voxelOpacities, const float3 camera_location, uchar3 * pixels, const int width, const int height);
	void* sr_args[] = { &dev_ray_directions, &dev_voxel_grid, &camera_location, &dev_pixels, &draw_width, &draw_height };
	cudaError_t cuda_status = cudaLaunchKernel((const void*)simulate_ray, grid_size, block_size, sr_args);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "simulate_ray launch failed: %s\n", cudaGetErrorString(cuda_status));
		return cuda_status;
	}
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simulate_ray!\n", cuda_status);
	return cuda_status;
}

cudaError_t load_device_memory(array<SimulateVoxelCube::Point3D, 3>^* voxel_grid, float3* dev_voxel_grid, int voxel_grid_size)
{
	const pin_ptr<SimulateVoxelCube::Point3D> voxel_grid_start = &(*voxel_grid)[0,0,0];
	cudaError_t cuda_status = cudaMemcpy(dev_voxel_grid, voxel_grid_start,
		voxel_grid_size * voxel_grid_size * voxel_grid_size * sizeof(float3),
		cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");
	return cuda_status;
}

cudaError_t retrieve_pixels(const uchar3* dev_pixels, interior_ptr<array<unsigned char>^> pixels, int draw_width, int draw_height)
{
	const pin_ptr<unsigned char> pixels_start = &(*pixels)[0];
	const cudaError_t cuda_status = cudaMemcpy(pixels_start, dev_pixels,
	                                           draw_width * draw_height * sizeof(uchar3),
	                                           cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");
	return cuda_status;
}

void free_resources(float3* dev_ray_directions, float* dev_ray_lengths, uchar3* dev_pixels, float3* dev_voxel_grid)
{
	cudaFree(dev_ray_directions);
	cudaFree(dev_ray_lengths);
	cudaFree(dev_pixels);
	cudaFree(dev_voxel_grid);
}

void free_2d_resources(float3* dev_ray_directions, float* dev_ray_lengths, uchar3* dev_pixels)
{
	cudaFree(dev_ray_directions);
	cudaFree(dev_ray_lengths);
	cudaFree(dev_pixels);
}

float operator !(const float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

float3 operator ^(const float3 v1, const float3 v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

float operator &(const float3 v1, const float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

float3 operator /(const float3 v, const float s)
{
	return make_float3(v.x / s, v.y / s, v.z / s);
}

float3 operator *(const float3 v, const float s)
{
	return make_float3(v.x * s, v.y * s, v.z * s);
}

float3 operator +(const float3 v1, const float3 v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

float3 operator -(const float3 v1, const float3 v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
float3 log(float3 v)
{
	return make_float3(logf(v.x), logf(v.y), logf(v.z));
}
//cast float3 to point3D

SimulateVoxelCube::Point3D asPoint3D(float3 v){
	return SimulateVoxelCube::Point3D(v.x, v.y, v.z);
};
float3 rotate_vec(const float3 vec, const float3 axis, const float cos, const float sin)
{
	return axis * ((1 - cos) * (axis & vec)) + vec * cos + (axis ^ vec) * sin;
}
float sqr(float f) { return f * f; }
dim3 block_size_2d = { 32, 32 };
dim3 block_size_3d = { 8,8,16 };
float3 get_density(float x, float y, float z)
{
	float pi = 3.14159265358979;
	//float i = 2*x - 1;
	//float j = 2*y - 1;
	//float k = 2*z - 1;
	//float f = 100 - (i * i + j * j + k * k) * 10000;
	//f = f > 1 ? 1 : f < 0 ? 0 : f;
	//return float3{ 0,0,f };
	float repetitions = 5;
	return float3{ sinf(repetitions * pi * x) * sinf(repetitions * pi * y) * sinf(repetitions * pi * z) / 2 + 0.5f, cosf(repetitions * pi * x) * cosf(repetitions * pi * y) * cosf(repetitions * pi * z) / 2 + 0.5f, 0.5f - sinf(repetitions * pi * x) * sinf(repetitions * pi * y) * sinf(repetitions * pi * z) / 2 };
}
namespace SimulateVoxelCube
{
	VoxelGridRenderer::VoxelGridRenderer()
	{
		if (cudaSetDevice(0) != cudaSuccess)
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		pixel_values = gcnew array<unsigned char>(draw_width * draw_height * 3);
		bitmap = gcnew Bitmap(draw_width, draw_height, draw_width * 3,
			PixelFormat::Format24bppRgb, Marshal::UnsafeAddrOfPinnedArrayElement(pixel_values, 0));
		array<Point3D, 3>^ voxel_grid = gcnew array<Point3D, 3>(voxel_grid_size, voxel_grid_size, voxel_grid_size);
		camera_relative_x = new float3{ sqrtf(1/2.f), sqrtf(1/2.f), 0.0f};
		camera_relative_y = new float3{ sqrtf(1/6.f), -sqrtf(1/6.f), sqrtf(2/3.f)};
		camera_relative_z = new float3{ sqrtf(1/3.f), -sqrtf(1/3.f), -sqrtf(1 / 3.f) };
		camera_location = new float3{camera_relative_z->x * -1 *voxel_grid_size, camera_relative_z->y * -1 * voxel_grid_size, camera_relative_z->z * -1 * voxel_grid_size};
		for(int i = 0; i < voxel_grid_size; i++)
			for (int j = 0; j < voxel_grid_size; j++)
				for (int k = 0; k < voxel_grid_size; k++)
					voxel_grid[i, j, k] = asPoint3D(log(float3{ 1,1,1 } - get_density(i / (float)voxel_grid_size, j / (float)voxel_grid_size, k / (float)voxel_grid_size)*0.999f));
		float3* dev_ray_directions_duplicate = dev_ray_directions;
		float* dev_ray_lengths_duplicate = dev_ray_lengths;
		uchar3* dev_pixels_duplicate = dev_pixels;
		float3* dev_voxel_grid_duplicate = dev_voxel_grid;
		alloc_device_memory(&dev_ray_directions_duplicate, &dev_ray_lengths_duplicate, &dev_pixels_duplicate, &dev_voxel_grid_duplicate, draw_width, draw_height, voxel_grid_size);
		dev_ray_directions = dev_ray_directions_duplicate;
		dev_ray_lengths = dev_ray_lengths_duplicate;
		dev_pixels = dev_pixels_duplicate;
		dev_voxel_grid = dev_voxel_grid_duplicate;
		load_device_memory(&voxel_grid, dev_voxel_grid, voxel_grid_size);
		dim3 grid_size = { (draw_width + block_size_2d.x - 1) / block_size_2d.x, (draw_height + block_size_2d.y - 1) / block_size_2d.y };
		get_direction_lengths(grid_size, block_size_2d, &dev_ray_lengths_duplicate, 1.0f, draw_width, draw_height);
		get_directions(grid_size, block_size_2d, &dev_ray_directions_duplicate, &dev_ray_lengths_duplicate, *camera_relative_x, *camera_relative_y, *camera_relative_z, 1.0f, draw_width, draw_height);
	}

	void VoxelGridRenderer::SetDrawSize(Size newSize)
	{
		draw_width = newSize.Width/4*4;
		draw_height = newSize.Height/4*4;
		pixel_values = gcnew array<unsigned char>(draw_width * draw_height * 3);
		bitmap = gcnew Bitmap(draw_width, draw_height, draw_width * 3,
			PixelFormat::Format24bppRgb, Marshal::UnsafeAddrOfPinnedArrayElement(pixel_values, 0));
		free_2d_resources(dev_ray_directions, dev_ray_lengths, dev_pixels);
		float3* dev_ray_directions_duplicate = dev_ray_directions;
		float* dev_ray_lengths_duplicate = dev_ray_lengths;
		uchar3* dev_pixels_duplicate = dev_pixels;
		alloc_2d_device_memory(&dev_ray_directions_duplicate, &dev_ray_lengths_duplicate, &dev_pixels_duplicate, draw_width, draw_height);
		dev_ray_directions = dev_ray_directions_duplicate;
		dev_ray_lengths = dev_ray_lengths_duplicate;
		dev_pixels = dev_pixels_duplicate;
		dim3 grid_size = { (draw_width + block_size_2d.x - 1) / block_size_2d.x, (draw_height + block_size_2d.y - 1) / block_size_2d.y };
		get_direction_lengths(grid_size, block_size_2d, &dev_ray_lengths_duplicate, 1.0f, draw_width, draw_height);
		get_directions(grid_size, block_size_2d, &dev_ray_directions_duplicate, &dev_ray_lengths_duplicate, *camera_relative_x, *camera_relative_y, *camera_relative_z, 1.0f, draw_width, draw_height);
	}


	Void VoxelGridRenderer::Render()
	{
		dim3 grid_size = { (draw_width + block_size_2d.x - 1) / block_size_2d.x, (draw_height + block_size_2d.y - 1) / block_size_2d.y };
		simulate_voxel_cube(grid_size, block_size_2d, dev_ray_directions, dev_voxel_grid, dev_pixels, *camera_location, draw_width, draw_height);
		retrieve_pixels(dev_pixels, &pixel_values, draw_width, draw_height);
	}


	Void VoxelGridRenderer::UpdateCameraLocation(Point3D location)
	{
		camera_location = new float3{ location.X, location.Y, location.Z };
	}

	Void VoxelGridRenderer::UpdateCameraOrientation(Point3D relativeX, Point3D relativeY, Point3D relativeZ)
	{
		camera_relative_x = new float3{ relativeX.X, relativeX.Y, relativeX.Z };
		camera_relative_y = new float3{ relativeY.X, relativeY.Y, relativeY.Z };
		camera_relative_z = new float3{ relativeZ.X, relativeZ.Y, relativeZ.Z };
		float* dev_ray_lengths_duplicate = dev_ray_lengths;
		float3* dev_ray_directions_duplicate = dev_ray_directions;
		dim3 block_size = { 32, 32, 1 };
		dim3 grid_size = { (draw_width + block_size.x - 1) / block_size.x, (draw_height + block_size.y - 1) / block_size.y };
		get_directions(grid_size, block_size, &dev_ray_directions_duplicate, &dev_ray_lengths_duplicate, *camera_relative_x, *camera_relative_y, *camera_relative_z, 1.0f, draw_width, draw_height);
		dev_ray_lengths = dev_ray_lengths_duplicate;
		dev_ray_directions = dev_ray_directions_duplicate;
	}

	Point3D VoxelGridRenderer::GetCameraLocation()
	{
		return Point3D{ camera_location->x, camera_location->y, camera_location->z };
	}

	Point3D VoxelGridRenderer::GetCameraRelativeX()
	{
		return Point3D{ camera_relative_x->x, camera_relative_x->y, camera_relative_x->z };
	}

	Point3D VoxelGridRenderer::GetCameraRelativeY()
	{
		return Point3D{ camera_relative_y->x, camera_relative_y->y, camera_relative_y->z };
	}

	Point3D VoxelGridRenderer::GetCameraRelativeZ()
	{
		return Point3D{ camera_relative_z->x, camera_relative_z->y, camera_relative_z->z };
	}

	VoxelGridRenderer::~VoxelGridRenderer()
	{
		free_all();
	}

	void VoxelGridRenderer::free_all()
	{
		free_resources(dev_ray_directions, dev_ray_lengths, dev_pixels, dev_voxel_grid);
	}
}