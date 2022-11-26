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
        private YoloV5 yolov5 { get; set; }
        private Bitmap[] defaultBmps { get; set; }
        private int currentBmp { get; set; }
        private Dictionary<int, string> labels { get; set; }

        public Main()
        {
            InitializeComponent();
            
            using (FileStream modelStream = new FileStream("./weights/yolov5s.cuda.pt", FileMode.Open))
            {
                yolov5 = new YoloV5(modelStream, YoloV5.IsCudaAvailable, false, 640, 640, 0.25f, 0.45f);
            }
            yolov5.WarmUp();
            defaultBmps = new Bitmap[2];
            defaultBmps[0] = new Bitmap("./images/bus.jpg");
            defaultBmps[1] = new Bitmap("./images/zidane.jpg");

            labels = DefaultLabels;
        }

        private static void DrawResult(Bitmap bmp, YoloResult[] results, Dictionary<int, string> labels)
        {
            using (var g = Graphics.FromImage(bmp))
            {
                foreach (var result in results)
                {
                    g.DrawRectangle(new Pen(Brushes.Red, 2), new Rectangle(result.X, result.Y, result.Width, result.Height));
                    string label = labels[result.ClassIndex];
                    string score = (result.Confidence * 100).ToString("#.##");
                    g.DrawString($"{label} ({score})", new Font(DefaultFont.FontFamily, 20, FontStyle.Bold),
                        Brushes.Red, new Point(result.X, result.Y - 25));
                }
            }
        }

        private void testBtn_Click(object sender, EventArgs e)
        {
            Bitmap bmp = defaultBmps[currentBmp];
            var result = yolov5.Predict(bmp);

            Bitmap newBmp = (Bitmap)bmp.Clone();
            DrawResult(newBmp, result, labels);
            pictureBox.Image = newBmp;

            currentBmp++;
            if (currentBmp >= defaultBmps.Length)
                currentBmp = 0;
        }

        private void inputBtn_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image Files (*.bmp;*.jpg;*.jpeg,*.png)|*.BMP;*.JPG;*.JPEG;*.PNG";
            ofd.Title = "Select Image";
            ofd.CheckFileExists = true;
            if (ofd.ShowDialog() != DialogResult.OK) return;
            Bitmap bmp;
            try
            {
                bmp = (Bitmap)Bitmap.FromFile(ofd.FileName);
            }
            catch
            {
                MessageBox.Show("Fail to open image");
                return;
            }
            var result = yolov5.Predict(bmp);
            DrawResult(bmp, result, labels);
            pictureBox.Image = bmp;
        }

        private static Dictionary<int, string> DefaultLabels
        {
            get
            {
                var result = new Dictionary<int, string>();
                result[0] = "person";
                result[1] = "bicycle";
                result[2] = "car";
                result[3] = "motorcycle";
                result[4] = "airplane";
                result[5] = "bus";
                result[6] = "train";
                result[7] = "truck";
                result[8] = "boat";
                result[9] = "traffic light";
                result[10] = "fire hydrant";
                result[11] = "stop sign";
                result[12] = "parking meter";
                result[13] = "bench";
                result[14] = "bird";
                result[15] = "cat";
                result[16] = "dog";
                result[17] = "horse";
                result[18] = "sheep";
                result[19] = "cow";
                result[20] = "elephant";
                result[21] = "bear";
                result[22] = "zebra";
                result[23] = "giraffe";
                result[24] = "backpack";
                result[25] = "umbrella";
                result[26] = "handbag";
                result[27] = "tie";
                result[28] = "suitcase";
                result[29] = "frisbee";
                result[30] = "skis";
                result[31] = "snowboard";
                result[32] = "sports ball";
                result[33] = "kite";
                result[34] = "baseball bat";
                result[35] = "baseball glove";
                result[36] = "skateboard";
                result[37] = "surfboard";
                result[38] = "tennis racket";
                result[39] = "bottle";
                result[40] = "wine glass";
                result[41] = "cup";
                result[42] = "fork";
                result[43] = "knife";
                result[44] = "spoon";
                result[45] = "bowl";
                result[46] = "banana";
                result[47] = "apple";
                result[48] = "sandwich";
                result[49] = "orange";
                result[50] = "broccoli";
                result[51] = "carrot";
                result[52] = "hot dog";
                result[53] = "pizza";
                result[54] = "donut";
                result[55] = "cake";
                result[56] = "chair";
                result[57] = "couch";
                result[58] = "potted plant";
                result[59] = "bed";
                result[60] = "dining table";
                result[61] = "toilet";
                result[62] = "tv";
                result[63] = "laptop";
                result[64] = "mouse";
                result[65] = "remote";
                result[66] = "keyboard";
                result[67] = "cell phone";
                result[68] = "microwave";
                result[69] = "oven";
                result[70] = "toaster";
                result[71] = "sink";
                result[72] = "refrigerator";
                result[73] = "book";
                result[74] = "clock";
                result[75] = "vase";
                result[76] = "scissors";
                result[77] = "teddy bear";
                result[78] = "hair drier";
                result[79] = "toothbrush";
                return result;
            }
        }
    }
}
