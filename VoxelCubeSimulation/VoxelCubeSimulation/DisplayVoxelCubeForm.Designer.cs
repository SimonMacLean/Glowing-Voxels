namespace VoxelCubeSimulation
{
    partial class DisplayVoxelCubeForm
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            SuspendLayout();
            // 
            // DisplayVoxelCubeForm
            // 
            AutoScaleDimensions = new SizeF(8F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = Color.Black;
            ClientSize = new Size(1000, 1000);
            DoubleBuffered = true;
            ForeColor = SystemColors.HighlightText;
            Name = "DisplayVoxelCubeForm";
            Text = "Form1";
            ClientSizeChanged += DisplayVoxelCubeForm_ClientSizeChanged;
            Paint += DisplayVoxelCubeForm_Paint;
            MouseDown += DisplayVoxelCubeForm_MouseDown;
            MouseMove += DisplayVoxelCubeForm_MouseMove;
            MouseUp += DisplayVoxelCubeForm_MouseUp;
            MouseWheel += DisplayVoxelCubeForm_MouseWheel;
            ResumeLayout(false);
        }

        #endregion
    }
}
