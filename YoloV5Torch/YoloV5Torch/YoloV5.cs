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
        [DllImport("YoloV5TorchCpp.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void TestMain();

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "TorchCudaIsAvailable", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool TorchCudaIsAvailable();

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "TorchCudaCudnnIsAvailable", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool TorchCudaCudnnIsAvailable();

        [DllImport("YoloV5TorchCpp.dll", EntryPoint = "TorchCudaDeviceCount", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool TorchCudaDeviceCount();

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

        public static bool IsCudaAvailable => TorchCudaIsAvailable();
        public static bool IsCudnnAvailable => TorchCudaCudnnIsAvailable();
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
        public YoloV5(string torchscriptPath,
            bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
        {
            Initialize(isCuda, isHalf, height, width, confThres, iouThres);
            this.Ptr = YoloV5New(torchscriptPath, isCuda, isHalf, height, width, confThres, iouThres);
        }

        public YoloV5(byte[] torchScriptArr,
            bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
        {
            Initialize(isCuda, isHalf, height, width, confThres, iouThres);
            this.Ptr = YoloV5New(torchScriptArr, torchScriptArr.Length, isCuda, isHalf, height, width, confThres, iouThres);
        }

        public YoloV5(Stream stream,
            bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
        {
            Initialize(isCuda, isHalf, height, width, confThres, iouThres);
            byte[] bytes = ReadAllBytes(stream);
            this.Ptr = YoloV5New(bytes, bytes.Length, isCuda, isHalf, height, width, confThres, iouThres);
        }

        public void Initialize(bool isCuda, bool isHalf, int height, int width, float confThres, float iouThres)
        {
            this.IsCuda = isCuda;
            this.IsHalf = isHalf;
            this.Height = height;
            this.Width = width;
            this.ConfThres = confThres;
            this.IouThres = iouThres;
        }

        public void WarmUp()
        {
            Bitmap bmp = new Bitmap(Height, Width);
            for (int i = 0; i < 3; i++)
            {
                this.Predict(bmp);
            }
        }

        private byte[] ReadAllBytes(Stream stream)
        {
            byte[] buffer = new byte[stream.Length];
            stream.Read(buffer, 0, (int)stream.Length);
            return buffer;
        }

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

        public YoloResult[][] Predicts(IEnumerable<Bitmap> bitmaps)
        {
            IntPtr[] mats = new IntPtr[bitmaps.Count()];

            int count = 0;
            for (int i = 0; i < bitmaps.Count(); i++)
            {
                IntPtr matPtr = IntPtr.Zero;
                OpenCv.BitmapToMatPtr(bitmaps.ElementAt(i), out matPtr);
                mats[count] = matPtr;
                count++;
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

            [DllImport("YoloV5TorchCpp.dll", EntryPoint = "Cv2GetMatDataAndInfo", CharSet = CharSet.Auto)]
            private static extern IntPtr GetMatDataAndInfo(IntPtr matPtr, ref int w, ref int h, ref int channel, ref int type);

            [DllImport("YoloV5TorchCpp.dll", EntryPoint = "Cv2DeleteMat", CharSet = CharSet.Auto)]
            public static extern IntPtr DeleteMat(IntPtr matPtr);


            public static Bitmap GetBitmapFromMatPtr(IntPtr matPtr)
            {
                int w = 0;
                int h = 0;
                int channel = 0;
                int type = 0;
                IntPtr ptr = GetMatDataAndInfo(matPtr, ref w, ref h, ref channel, ref type);
                int stride = channel * w;
                PixelFormat format;
                switch (type)
                {
                    case 1:
                        format = PixelFormat.Format8bppIndexed;
                        break;
                    case 3:
                        format = PixelFormat.Format24bppRgb;
                        break;
                    default:
                        return null;
                }

                return GetBitmapByData(ptr, w, h, stride, format);
            }

            public static Bitmap GetBitmapByData(IntPtr ptr, int w, int h, int stride, PixelFormat format)
            {

                int rowBytes = w * Image.GetPixelFormatSize(format) / 8;
                byte[] rgbValues = new byte[stride * h];
                for (var i = 0; i < h; i++)
                {

                    Marshal.Copy(ptr, rgbValues, i * stride, rowBytes);
                    ptr += rowBytes; // next row
                }

                GCHandle hObject = GCHandle.Alloc(rgbValues, GCHandleType.Pinned);
                IntPtr ptrnew = hObject.AddrOfPinnedObject();

                Bitmap bitmap = new Bitmap(w, h, stride, format, ptrnew);

                if (format == PixelFormat.Format8bppIndexed)
                {
                    //bitmap.Palette = GrayPalette;
                }

                hObject.Free();
                return bitmap;
            }


            public static void BitmapToMatPtr(Bitmap bitmap, out IntPtr matPtr)
            {
                int stride;
                int channel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;
                byte[] source = GetBGRValues(bitmap, out stride);
                GetMat(source, bitmap.Width, bitmap.Height, channel, out matPtr);
            }

            public static byte[] GetBGRValues(Bitmap bmp, out int stride)
            {
                var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
                var bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, bmp.PixelFormat);
                stride = bmpData.Stride;

                var rowBytes = bmpData.Width * Image.GetPixelFormatSize(bmp.PixelFormat) / 8;
                var imgBytes = bmp.Height * rowBytes;
                byte[] rgbValues = new byte[imgBytes];

                int height = bmp.Height;
                int _stride = stride;
                IntPtr ptr = bmpData.Scan0;

                Parallel.For(0, height, i =>
                {
                    IntPtr _ptr = new IntPtr(ptr.ToInt64() + i * _stride);
                    Marshal.Copy(_ptr, rgbValues, i * rowBytes, rowBytes);
                });
                bmp.UnlockBits(bmpData);

                return rgbValues;
            }

        }
    }
}
