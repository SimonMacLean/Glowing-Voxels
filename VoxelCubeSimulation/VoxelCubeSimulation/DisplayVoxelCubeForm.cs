using ReactionDiffusionSimulation;
using SimulateVoxelCube;
using Timer = System.Windows.Forms.Timer;

namespace VoxelCubeSimulation
{
    public partial class DisplayVoxelCubeForm : Form
    {
        private readonly VoxelGridRenderer _voxelGridRenderer;
        private readonly ReactionDiffusionSimulator _reactionDiffusionSimulator;
        private Point _mousePosition;
        private bool _mouseDown = false;
        private readonly Timer _t = new()
        {
            Enabled = true,
            Interval = 1000 / 60
        };
        private float _kill;
        private float _feed;

        public DisplayVoxelCubeForm()
        {
            InitializeComponent();
            _voxelGridRenderer = new VoxelGridRenderer();
            _reactionDiffusionSimulator = new ReactionDiffusionSimulator();
            _t.Tick += T_Tick;
            _kill = killSlider.Value / 1000f;
            _feed = feedSlider.Value / 1000f;
        }

        private void T_Tick(object? sender, EventArgs e)
        {
           unsafe
           {
              // _reactionDiffusionSimulator.SimulateFrame(_voxelGridRenderer.dev_voxel_grid, _feed, _kill);
           }
           Invalidate();
        }

        private void DisplayVoxelCubeForm_MouseDown(object sender, MouseEventArgs e)
        {
            _mousePosition = e.Location;
            _mouseDown = true;
        }

        Point3D RotatePoint(Point3D point, Point3D axis, float angle)
        {
            float angleSin = (float)Math.Sin(angle);
            float angleCos = (float)Math.Cos(angle);
            float d = (1 - angleCos) * (axis.X * point.X + axis.Y * point.Y + axis.Z * point.Z);
            Point3D cross = new(axis.Y * point.Z - axis.Z * point.Y, axis.Z * point.X - axis.X * point.Z, axis.X * point.Y - axis.Y * point.X);
            return new Point3D(
                axis.X * d + point.X * angleCos + cross.X * angleSin,
                axis.Y * d + point.Y * angleCos + cross.Y * angleSin,
                axis.Z * d + point.Z * angleCos + cross.Z * angleSin
            );
        }

        private static float Length(Point3D point3D)
        {
            return MathF.Sqrt(point3D.X * point3D.X + point3D.Y * point3D.Y + point3D.Z * point3D.Z);
        }

        private void DisplayVoxelCubeForm_Paint(object sender, PaintEventArgs e)
        {
            _voxelGridRenderer.Render();
            e.Graphics.DrawImage(_voxelGridRenderer.bitmap, 0, 0);
        }

        private void DisplayVoxelCubeForm_MouseMove(object sender, MouseEventArgs e)
        {
            if (_mouseDown)
            {
                PointF offset = new PointF((e.Location.X - _mousePosition.X) * 0.01f, (e.Location.Y - _mousePosition.Y) * 0.01f);
                Point3D cameraRelativeX = _voxelGridRenderer.GetCameraRelativeX();
                Point3D cameraRelativeY = _voxelGridRenderer.GetCameraRelativeY();
                Point3D cameraRelativeZ = _voxelGridRenderer.GetCameraRelativeZ();
                Point3D cameraLocation = _voxelGridRenderer.GetCameraLocation();
                cameraRelativeX = RotatePoint(cameraRelativeX, cameraRelativeY, offset.X);
                cameraRelativeZ = RotatePoint(cameraRelativeZ, cameraRelativeY, offset.X);
                cameraLocation = RotatePoint(cameraLocation, cameraRelativeY, offset.X);
                cameraRelativeY = RotatePoint(cameraRelativeY, cameraRelativeX, offset.Y);
                cameraRelativeZ = RotatePoint(cameraRelativeZ, cameraRelativeX, offset.Y);
                cameraLocation = RotatePoint(cameraLocation, cameraRelativeX, offset.Y);
                cameraRelativeX = new Point3D(cameraRelativeX.X / Length(cameraRelativeX), cameraRelativeX.Y / Length(cameraRelativeX), cameraRelativeX.Z / Length(cameraRelativeX));
                cameraRelativeY = new Point3D(cameraRelativeY.X / Length(cameraRelativeY), cameraRelativeY.Y / Length(cameraRelativeY), cameraRelativeY.Z / Length(cameraRelativeY));
                cameraRelativeZ = new Point3D(cameraRelativeZ.X / Length(cameraRelativeZ), cameraRelativeZ.Y / Length(cameraRelativeZ), cameraRelativeZ.Z / Length(cameraRelativeZ));
                _voxelGridRenderer.UpdateCameraLocation(cameraLocation);
                _voxelGridRenderer.UpdateCameraOrientation(cameraRelativeX, cameraRelativeY, cameraRelativeZ);
                Invalidate();
            }
            _mousePosition = e.Location;
        }

        private void DisplayVoxelCubeForm_MouseWheel(object sender, MouseEventArgs e)
        {
            if (_mousePosition.X < 0 || _mousePosition.Y < 0 || _mousePosition.X >= ClientSize.Width || _mousePosition.Y >= ClientSize.Height)
                return;
            Point3D cameraRelativeX = _voxelGridRenderer.GetCameraRelativeX();
            Point3D cameraRelativeY = _voxelGridRenderer.GetCameraRelativeY();
            Point3D cameraRelativeZ = _voxelGridRenderer.GetCameraRelativeZ();
            Point3D cameraLocation = _voxelGridRenderer.GetCameraLocation();
            const float focalLength = 1;
            const float scrollWheelMovementSize = 10;
            float dx = (2.0f * _mousePosition.X) / ClientSize.Width - 1;
            float dy = 1 - (2.0f * _mousePosition.Y) / ClientSize.Height;
            Point3D mouseDiff = new(cameraRelativeZ.X * focalLength + cameraRelativeX.X * dx + cameraRelativeY.X * dy,
                cameraRelativeZ.Y * focalLength + cameraRelativeX.Y * dx + cameraRelativeY.Y * dy,
                cameraRelativeZ.Z * focalLength + cameraRelativeX.Z * dx + cameraRelativeY.Z * dy);
            mouseDiff = new Point3D(mouseDiff.X / Length(mouseDiff) * (scrollWheelMovementSize * e.Delta / 100f),
                mouseDiff.Y / Length(mouseDiff) * (scrollWheelMovementSize * e.Delta / 100f),
                mouseDiff.Z / Length(mouseDiff) * (scrollWheelMovementSize * e.Delta / 100f));
            cameraLocation = new Point3D(cameraLocation.X + mouseDiff.X, cameraLocation.Y + mouseDiff.Y, cameraLocation.Z + mouseDiff.Z);
            _voxelGridRenderer.UpdateCameraLocation(cameraLocation);
            Invalidate();
        }

        private void DisplayVoxelCubeForm_MouseUp(object sender, MouseEventArgs e) => _mouseDown = false;


        private void DisplayVoxelCubeForm_ClientSizeChanged(object sender, EventArgs e)
        {
            _voxelGridRenderer.SetDrawSize(ClientSize);

            Invalidate();
        }

        private void killSlider_Scroll(object sender, EventArgs e)
        {
            _kill = killSlider.Value / 1000f;
        }

        private void feedSlider_Scroll(object sender, EventArgs e)
        {
            _feed = feedSlider.Value / 1000f;
        }
    }
}
