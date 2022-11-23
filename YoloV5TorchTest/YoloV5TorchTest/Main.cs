using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using YoloV5Torch;
using System.IO;

namespace YoloV5TorchTest
{
    public partial class Main : Form
    {
        public Main()
        {
            InitializeComponent();
            FileStream modelStream = new FileStream("./weights/yolov5s.cuda.pt", FileAccess.Read);

        }
    }
}
