using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Drawing.Imaging;

namespace YoloV5Torch
{
    [StructLayout(LayoutKind.Sequential)]
    public struct YoloResult
    {
        public int ClassIndex;
        public float Confidence;
        public int X;
        public int Y;
        public int Width;
        public int Height;
    }

    public class YoloV5 : IDisposable
    {
        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "TorchCudaIsAvailable", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool TorchCudaIsAvailable();

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "TorchCudaCudnnIsAvailable", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool TorchCudaCudnnIsAvailable();

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "TorchCudaDeviceCount", CallingConvention = CallingConvention.Cdecl)]
        private static extern int TorchCudaDeviceCount();

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "TorchVersion", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GetTorchVersion(out IntPtr str, out int length);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "CStrDelete", CallingConvention = CallingConvention.Cdecl)]
        private static extern void CStrDelete(IntPtr cstr);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5NewByPath", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr YoloV5New([MarshalAs(UnmanagedType.LPStr)] string torchscriptPath,
            bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5NewByArray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr YoloV5New(byte[] torchScriptArr, int torchScriptLength,
            bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5Delete", CallingConvention = CallingConvention.Cdecl)]
        private static extern void YoloV5Delete(IntPtr yolov5);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5Preditct", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr YoloV5Preditct(IntPtr yolov5, IntPtr cvMat);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5Preditcts", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr YoloV5Preditcts(IntPtr yolov5, IntPtr[] matArr, int matArrLength);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5ResultAt", CallingConvention = CallingConvention.Cdecl)]
        private static extern YoloResult YoloV5ResultAt(IntPtr result, int index);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5ResultSize", CallingConvention = CallingConvention.Cdecl)]
        private static extern int YoloV5ResultSize(IntPtr result);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5ResultDelete", CallingConvention = CallingConvention.Cdecl)]
        private static extern void YoloV5ResultDelete(IntPtr result);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5ResultsAt", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr YoloV5ResultsAt(IntPtr results, int index);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5ResultsSize", CallingConvention = CallingConvention.Cdecl)]
        private static extern int YoloV5ResultsSize(IntPtr results);

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "YoloV5ResultsDelete", CallingConvention = CallingConvention.Cdecl)]
        private static extern void YoloV5ResultsDelete(IntPtr results);

        public IntPtr Ptr { get; private set; }
        public bool IsCuda { get; private set; }
        public bool IsHalf { get; private set; }
        public int Height { get; private set; }
        public int Width { get; private set; }
        public float ConfThres { get; private set; }
        public float IouThres { get; private set; }

        /// <summary>
        /// is cuda available
        /// </summary>
        public static bool IsCudaAvailable => TorchCudaIsAvailable();
        /// <summary>
        /// is cudnn available
        /// </summary>
        public static bool IsCudnnAvailable => TorchCudaCudnnIsAvailable();
        /// <summary>
        /// Number of cuda devices
        /// </summary>
        public static int CudaDeviceCount => TorchCudaDeviceCount();
        /// <summary>
        /// torch version
        /// </summary>
        public static string TorchVersion
        {
            get
            {
                GetTorchVersion(out IntPtr strPtr, out int length);
                string result = Marshal.PtrToStringAnsi(strPtr, length);
                CStrDelete(strPtr);
                return result;
            }
        }
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="torchscriptPath">path of torchscript</param>
        /// <param name="isCuda">is using cuda</param>
        /// <param name="isHalf">is half precision</param>
        /// <param name="height">height of model</param>
        /// <param name="width">width of model</param>
        /// <param name="confThres">confidence threshold</param>
        /// <param name="iouThres">iou threshold</param>
        public YoloV5(string torchscriptPath,
            bool isCuda, bool isHalf = false, int height = 640, int width = 640, float confThres = 0.25f, float iouThres = 0.45f)
        {
            Initialize(isCuda, isHalf, height, width, confThres, iouThres);
            this.Ptr = YoloV5New(torchscriptPath, isCuda, isHalf, height, width, confThres, iouThres);
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="torchScriptArr">bytes of torchscript</param>
        /// <param name="isCuda">is using cuda</param>
        /// <param name="isHalf">is half precision</param>
        /// <param name="height">height of model</param>
        /// <param name="width">width of model</param>
        /// <param name="confThres">confidence threshold</param>
        /// <param name="iouThres">iou threshold</param>
        public YoloV5(byte[] torchScriptArr,
            bool isCuda, bool isHalf = false, int height = 640, int width = 640, float confThres = 0.25f, float iouThres = 0.45f)
        {
            Initialize(isCuda, isHalf, height, width, confThres, iouThres);
            this.Ptr = YoloV5New(torchScriptArr, torchScriptArr.Length, isCuda, isHalf, height, width, confThres, iouThres);
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="stream">stream of torchscript</param>
        /// <param name="isCuda">is using cuda</param>
        /// <param name="isHalf">is half precision</param>
        /// <param name="height">height of model</param>
        /// <param name="width">width of model</param>
        /// <param name="confThres">confidence threshold</param>
        /// <param name="iouThres">iou threshold</param>
        public YoloV5(Stream stream,
            bool isCuda, bool isHalf = false, int height = 640, int width = 640, float confThres = 0.25f, float iouThres = 0.45f)
        {
            Initialize(isCuda, isHalf, height, width, confThres, iouThres);
            byte[] bytes = ReadAllBytes(stream);
            this.Ptr = YoloV5New(bytes, bytes.Length, isCuda, isHalf, height, width, confThres, iouThres);
        }

        private void Initialize(bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
        {
            this.IsCuda = isCuda;
            this.IsHalf = isHalf;
            this.Height = height;
            this.Width = width;
            this.ConfThres = confThres;
            this.IouThres = iouThres;
        }

        /// <summary>
        /// Warm up torch
        /// </summary>
        public void WarmUp()
        {
            using (Bitmap bmp = new Bitmap(Height, Width))
            {
                for (int i = 0; i < 3; i++)
                {
                    Predict(bmp);
                }
            }
        }

        private byte[] ReadAllBytes(Stream stream)
        {
            byte[] buffer = new byte[stream.Length];
            stream.Read(buffer, 0, (int)stream.Length);
            return buffer;
        }

        /// <summary>
        /// Predict by bitmap
        /// </summary>
        /// <param name="bitmap">bitmap</param>
        /// <returns>Prediction result of the bitmap</returns>
        public YoloResult[] Predict(Bitmap bitmap)
        {
            OpenCv.BitmapToMatPtr(bitmap, out IntPtr matPtr);
            IntPtr cppResults = YoloV5Preditct(Ptr, matPtr);
            OpenCv.DeleteMat(matPtr);

            int length = YoloV5ResultSize(cppResults);
            YoloResult[] result = new YoloResult[length];
            Parallel.For(0, length, i =>
            {
                var item = YoloV5ResultAt(cppResults, i);
                result[i] = item;
            });
            YoloV5ResultDelete(cppResults);
            return result;
        }

        /// <summary>
        /// Predict by bitmaps 
        /// </summary>
        /// <param name="bitmaps">bitmap collection</param>
        /// <returns>Prediction result of bitmap collection</returns>
        public YoloResult[][] Predicts(IEnumerable<Bitmap> bitmaps)
        {
            int bmpNums = bitmaps.Count();
            IntPtr[] mats = new IntPtr[bmpNums];

            for (int i = 0; i < bmpNums; i++)
            {
                IntPtr matPtr = IntPtr.Zero;
                OpenCv.BitmapToMatPtr(bitmaps.ElementAt(i), out matPtr);
                mats[i] = matPtr;
            }

            IntPtr cppResult = YoloV5Preditcts(Ptr, mats, mats.Length);
            foreach (IntPtr matPtr in mats)
            {
                OpenCv.DeleteMat(matPtr);
            }

            int resultLength = YoloV5ResultsSize(cppResult);
            YoloResult[][] results = new YoloResult[resultLength][];
            Parallel.For(0, resultLength, i =>
            {
                OpenCv.DeleteMat(mats[i]);
                IntPtr items = YoloV5ResultsAt(cppResult, i);
                int itemsLength = YoloV5ResultSize(items);
                results[i] = new YoloResult[itemsLength];
                for (int j = 0; j < itemsLength; j++)
                {
                    results[i][j] = YoloV5ResultAt(items, j);
                }
                YoloV5ResultDelete(items);
            });
            YoloV5ResultsDelete(cppResult);
            return results;
        }

        protected virtual void Dispose(bool bDisposing)
        {
            if (this.Ptr != IntPtr.Zero)
            {
                YoloV5Delete(this.Ptr);
                this.Ptr = IntPtr.Zero;
            }

            if (bDisposing)
            {
                GC.SuppressFinalize(this);
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }

        ~YoloV5()
        {
            Dispose(false);
        }

        private static class OpenCv
        {
            [DllImport("YoloV5TorchCpp.dll", EntryPoint = "Cv2GetMat", CharSet = CharSet.Auto)]
            private static extern void GetMat(byte[] src, int w, int h, int channel, out IntPtr intPtr);

            [DllImport("YoloV5TorchCpp.dll", EntryPoint = "Cv2GetMatData", CharSet = CharSet.Auto)]
            private static extern IntPtr GetMatData(IntPtr matPtr, out int w, out int h, out int channel, out int type);

            [DllImport("YoloV5TorchCpp.dll", EntryPoint = "Cv2DeleteMat", CharSet = CharSet.Auto)]
            public static extern void DeleteMat(IntPtr matPtr);

            [DllImport("YoloV5TorchCpp.dll", EntryPoint = "Cv2ShowMat", CharSet = CharSet.Auto)]
            public static extern void ShowMat([MarshalAs(UnmanagedType.LPStr)] string winName, IntPtr matPtr, int delay);


            public static Bitmap GetBitmapFromMatPtr(IntPtr matPtr)
            {
                int w, h, channel, type;
                IntPtr ptr = GetMatData(matPtr, out w, out h, out channel, out type);
                // stride must multiple of 4 
                int stride = channel * w;
                stride = stride + (4 - stride % 4);

                PixelFormat format;
                switch (type)
                {
                    case 1:
                        format = PixelFormat.Format8bppIndexed;
                        break;
                    case 2:
                        format = PixelFormat.Format16bppGrayScale;
                        break;
                    case 3:
                        format = PixelFormat.Format24bppRgb;
                        break;
                    case 4:
                        format = PixelFormat.Format32bppArgb;
                        break;
                    default:
                        return null;
                }

                return BitmapFromBytes(ptr, w, h, stride, format);
            }

            private static Bitmap BitmapFromBytes(IntPtr ptr, int w, int h, int stride, PixelFormat format)
            {
                int rowSize = w * Image.GetPixelFormatSize(format) / 8;

                // bytes store in Bitmap is in BGRA / BGRA format
                byte[] imgBytes = new byte[stride * h];
                Parallel.For(0, h, i =>
                {
                    IntPtr tempPtr = ptr + i * rowSize;
                    Marshal.Copy(tempPtr, imgBytes, i * stride, rowSize);
                });

                Bitmap bitmap = new Bitmap(w, h, stride, format, Marshal.UnsafeAddrOfPinnedArrayElement(imgBytes, 0));

                if (format == PixelFormat.Format8bppIndexed)
                {
                    //bitmap.Palette = GrayPalette;
                }

                return bitmap;
            }

            public static void BitmapToMatPtr(Bitmap bitmap, out IntPtr matPtr)
            {
                int stride;
                int channel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;
                byte[] source = GetBitmapBytes(bitmap, out stride);
                GetMat(source, bitmap.Width, bitmap.Height, channel, out matPtr);
            }

            private static byte[] GetBitmapBytes(Bitmap bmp, out int stride)
            {
                int width = bmp.Width;
                int height = bmp.Height;

                var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
                var bmpBytes = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, bmp.PixelFormat);
                stride = bmpBytes.Stride;

                var rowSize = width * Image.GetPixelFormatSize(bmp.PixelFormat) / 8;
                var bmpSize = height * rowSize;
                // bytes store in Bitmap is in BGRA / BGRA format
                byte[] imgBytes = new byte[bmpSize];

                IntPtr ptr = bmpBytes.Scan0;

                int tempStride = stride;
                Parallel.For(0, height, i =>
                {
                    IntPtr tempPtr = ptr + i * tempStride;
                    Marshal.Copy(tempPtr, imgBytes, i * rowSize, rowSize);
                });
                bmp.UnlockBits(bmpBytes);

                return imgBytes;
            }

        }
    }
}
