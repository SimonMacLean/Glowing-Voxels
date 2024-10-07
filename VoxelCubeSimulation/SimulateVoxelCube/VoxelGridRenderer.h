#pragma once
#include <cuda_runtime.h>
namespace SimulateVoxelCube {
	using namespace System;
	using namespace System::Drawing;
	using namespace Imaging;
	using namespace Reflection;
	using namespace Runtime::InteropServices;
	using namespace Windows::Forms;
	public value class Point3D
	{
	public:
		float X;
		float Y;
		float Z;
		Point3D(float x, float y, float z) : X(x), Y(y), Z(z) {}
	};

	public ref class VoxelGridRenderer
	{
	public:
		VoxelGridRenderer();
		Void Render();
		Void UpdateCameraLocation(Point3D location);
		Void UpdateCameraOrientation(Point3D relativeX, Point3D relativeY, Point3D relativeZ);
		Point3D GetCameraLocation();
		Point3D GetCameraRelativeX();
		Point3D GetCameraRelativeY();
		Point3D GetCameraRelativeZ();
		Void SetDrawSize(Size newSize);
		Bitmap^ bitmap;
		float3* dev_voxel_grid;
		~VoxelGridRenderer();

	private:
		void free_all();
		float* dev_ray_lengths;
		float3* dev_ray_directions;
		uchar3* dev_pixels;
		float3* camera_location;
		float3* camera_relative_x;
		float3* camera_relative_y;
		float3* camera_relative_z;
		array<unsigned char>^ pixel_values;
		Int32 draw_height = 1000;
		Int32 draw_width = 1000;
	};
}