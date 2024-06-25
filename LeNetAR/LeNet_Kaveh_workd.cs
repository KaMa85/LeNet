using UnityEngine;
using System;
using System.Linq;
using System.Collections;
using UnityEngine.Windows.WebCam;
using Microsoft.MixedReality.Toolkit;
using UnityEngine.Windows.Speech;
using System.Collections.Generic;

public class LeNet_Kaveh : MonoBehaviour
{
    public string text;
    PhotoCapture photoCaptureObject = null;
    Texture2D targetTexture = null;
    public float DTC;
    public Material OutputMaterial;
    public Texture2D StaticTexture;
    int n1, m1;
    public int pred;
    public bool isCollect = false;
    CameraParameters cameraParameters = new CameraParameters();
    Renderer rndr;
    Texture texture = null;
    KeywordRecognizer keywordRecognizer;
    Dictionary<string, System.Action> keywords = new Dictionary<string, System.Action>();
    void Start()
    {
        rndr = GetComponent<Renderer>();
        Resolution cameraResolution = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First();
        targetTexture = new Texture2D(cameraResolution.width, cameraResolution.height);

        // Create a PhotoCapture object
        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject)
        {
            photoCaptureObject = captureObject;
            cameraParameters.hologramOpacity = 0.5f;
            cameraParameters.cameraResolutionWidth = cameraResolution.width;
            cameraParameters.cameraResolutionHeight = cameraResolution.height;
            cameraParameters.pixelFormat = CapturePixelFormat.BGRA32;
            print(cameraResolution.width);
            print(cameraResolution.height);

            // Activate the camera
            photoCaptureObject.StartPhotoModeAsync(cameraParameters, delegate (PhotoCapture.PhotoCaptureResult result)
            {
            });
        });
        //Create keywords for keyword recognizer
        keywords.Add("collect", () =>
        {
            photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
            isCollect = true;
        });
        keywords.Add("remove", () =>
        {
            if (isCollect)
            {
                gameObject.GetComponent<Renderer>().enabled = false;
                isCollect = false;
            }
        });
        keywordRecognizer = new KeywordRecognizer(keywords.Keys.ToArray());
        keywordRecognizer.OnPhraseRecognized += KeywordRecognizer_OnPhraseRecognized;
        keywordRecognizer.Start();
    }
    private void Update()
    {
        

    }
    private void KeywordRecognizer_OnPhraseRecognized(PhraseRecognizedEventArgs args)
    {
        System.Action keywordAction;
        // if the keyword recognized is in our dictionary, call that Action.
        if (keywords.TryGetValue(args.text, out keywordAction))
        {
            keywordAction.Invoke();
        }
    }
    void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        // Copy the raw image data into our target texture
        photoCaptureFrame.UploadImageDataToTexture(targetTexture);
    RaycastHit hit;
        if (Physics.Raycast(Camera.main.transform.position, Camera.main.transform.forward, out hit, float.PositiveInfinity, LayerMask.GetMask("Spatial Awareness")))
        {
            float distanceToCrack = Vector3.Distance(hit.point, Camera.main.transform.position);
            DTC = distanceToCrack;
        }
        // duplicate the original texture and assign to the material
        // tint each mip level
        var cols = targetTexture.GetPixels32(0);
        n1 = (int)Mathf.Floor(targetTexture.height / 2 - 32 * 3 / 2);
        m1 = (int)Mathf.Floor(targetTexture.width/2 - 32 * 3 / 2);
        double[] grey_rect = new double[32 * 32];
        int ind_i=0, ind_j=0;


        for (int i = n1; i < n1 + 3 * 32; i += 3)
        {
            for (int j = m1; j < m1 + 3 * 32; j += 3)
            {
                grey_rect[ind_i + ind_j * 32] =
                    0.587 * (cols[i * targetTexture.width + j].g + cols[i * targetTexture.width + j - targetTexture.width - 1].g +
                    cols[i * targetTexture.width + j - targetTexture.width + 1].g + cols[i * targetTexture.width + j - 1].g + cols[i * targetTexture.width + j + 1].g +
                    cols[i * targetTexture.width + j + targetTexture.width - 1].g + cols[i * targetTexture.width + j + targetTexture.width + 1].g +
                    cols[i * targetTexture.width + j - targetTexture.width].g + cols[i * targetTexture.width + j + targetTexture.width].g) +
                    0.299 * (cols[i * targetTexture.width + j].r + cols[i * targetTexture.width + j - targetTexture.width - 1].r +
                    cols[i * targetTexture.width + j - targetTexture.width + 1].r + cols[i * targetTexture.width + j - 1].r + cols[i * targetTexture.width + j + 1].r +
                    cols[i * targetTexture.width + j + targetTexture.width - 1].r + cols[i * targetTexture.width + j + targetTexture.width + 1].r +
                    cols[i * targetTexture.width + j - targetTexture.width].r + cols[i * targetTexture.width + j + targetTexture.width].r) +
                    0.114 * (cols[i * targetTexture.width + j].b + cols[i * targetTexture.width + j - targetTexture.width - 1].b +
                    cols[i * targetTexture.width + j - targetTexture.width + 1].b + cols[i * targetTexture.width + j - 1].b + cols[i * targetTexture.width + j + 1].b +
                    cols[i * targetTexture.width + j + targetTexture.width - 1].b + cols[i * targetTexture.width + j + targetTexture.width + 1].b +
                    cols[i * targetTexture.width + j - targetTexture.width].b + cols[i * targetTexture.width + j + targetTexture.width].b);
                ind_j++;
            }
            ind_i++;
            ind_j = 0;
        }



        // First convolution (first layer i.e., image to second layer)
        double[] conv1_weight_0 = new double[]
        {
        -0.011, -0.1427, -0.1903, -0.3354, -0.0851, -0.1643, -0.1286, -0.3591, -0.0863, 0.1598, 0.084, -0.2931, -0.0535, 0.4418, 0.3627, 0.4161, 0.0728, 0.1063, 0.1174, -0.2518, 0.1052, 0.0885, 0.1357, 0.1211, -0.2999
        };

        double[] conv1_weight_1 = new double[]
        {
        -0.4011, -0.0836, -0.014, 0.0584, 0.1431, -0.3234, 0.1139, -0.0102, 0.2947, 0.1712, -0.3235, -0.0978, 0.383, 0.3269, 0.0879, -0.1618, 0.2292, 0.2021, 0.2099, -0.261, 0.0929, 0.1548, 0.0836, -0.1773, -0.292
        };

        double[] conv1_weight_2 = new double[]
        {
        -0.2868, 0.0213, 0.2315, 0.036, -0.0068, 0.3222, 0.0904, 0.0353, -0.1416, -0.1928, 0.0838, -0.2496, -0.2423, -0.0508, 0.1672, -0.2873, 0.1251, -0.0628, 0.275, 0.2391, 0.0813, 0.2482, -0.0053, -0.2134, -0.0545
        };

        double[] conv1_weight_3 = new double[]
        {
        -0.1733, 0.1119, -0.0191, -0.1138, -0.045, -0.0656, 0.1347, 0.1196, 0.0555, -0.191, 0.0416, 0.1583, -0.0066, -0.1445, -0.1753, 0.0438, 0.4356, 0.2995, -0.1321, -0.3908, 0.0914, 0.3277, 0.1445, 0.0319, -0.4282
        };

        double[] conv1_weight_4 = new double[]
        {
        -0.089, 0.3982, 0.0195, 0.0925, 0.2449, 0.1693, 0.2244, 0.4105, 0.1021, 0.3929, -0.0891, -0.0605, 0.1327, 0.2838, -0.0597, -0.3588, -0.444, -0.4219, -0.4696, -0.0646, -0.2252, -0.6029, -0.6996, -0.4065, -0.1719
        };

        double[] conv1_weight_5 = new double[]
        {
        0.0539, -0.1596, 0.1074, 0.1936, 0.074, -0.0058, -0.2195, -0.42, -0.1502, -0.128, 0.2611, -0.3387, -0.2727, -0.3027, -0.3124, 0.2238, 0.1067, -0.1114, -0.0899, 0.0248, 0.3108, 0.4757, 0.5179, 0.4184, 0.2931
        };

        double[] conv1_bias = new double[]
        {
        -0.0182, -0.1788, -0.2095, -0.0573, 0.137, -0.0677
        };
        double[] l2c0 = new double[28 * 28];
        double[] l2c1 = new double[28 * 28];
        double[] l2c2 = new double[28 * 28];
        double[] l2c3 = new double[28 * 28];
        double[] l2c4 = new double[28 * 28];
        double[] l2c5 = new double[28 * 28];

        for (int i = 2; i < 30; i++)
        {
            for (int j = 2; j < 30; j++)
            {
                int index = (i - 2) * 28 + (j - 2);
                int index_img = (i - 2) * 32 + (j - 2);

                l2c0[index] = Math.Max(
                    SumProduct(grey_rect, conv1_weight_0, index_img, 0, 5) +
                    SumProduct(grey_rect, conv1_weight_0, index_img + 32, 5, 5) +
                    SumProduct(grey_rect, conv1_weight_0, index_img + 64, 10, 5) +
                    SumProduct(grey_rect, conv1_weight_0, index_img + 96, 15, 5) +
                    SumProduct(grey_rect, conv1_weight_0, index_img + 128, 20, 5) + conv1_bias[0], 0);

                l2c1[index] = Math.Max(
                    SumProduct(grey_rect, conv1_weight_1, index_img, 0, 5) +
                    SumProduct(grey_rect, conv1_weight_1, index_img + 32, 5, 5) +
                    SumProduct(grey_rect, conv1_weight_1, index_img + 64, 10, 5) +
                    SumProduct(grey_rect, conv1_weight_1, index_img + 96, 15, 5) +
                    SumProduct(grey_rect, conv1_weight_1, index_img + 128, 20, 5) + conv1_bias[1], 0);
                l2c2[index] = Math.Max(
                    SumProduct(grey_rect, conv1_weight_2, index_img, 0, 5) +
                    SumProduct(grey_rect, conv1_weight_2, index_img + 32, 5, 5) +
                    SumProduct(grey_rect, conv1_weight_2, index_img + 64, 10, 5) +
                    SumProduct(grey_rect, conv1_weight_2, index_img + 96, 15, 5) +
                    SumProduct(grey_rect, conv1_weight_2, index_img + 128, 20, 5) + conv1_bias[2], 0);

                l2c3[index] = Math.Max(
                    SumProduct(grey_rect, conv1_weight_3, index_img, 0, 5) +
                    SumProduct(grey_rect, conv1_weight_3, index_img + 32, 5, 5) +
                    SumProduct(grey_rect, conv1_weight_3, index_img + 64, 10, 5) +
                    SumProduct(grey_rect, conv1_weight_3, index_img + 96, 15, 5) +
                    SumProduct(grey_rect, conv1_weight_3, index_img + 128, 20, 5) + conv1_bias[3], 0);

                l2c4[index] = Math.Max(
                    SumProduct(grey_rect, conv1_weight_4, index_img, 0, 5) +
                    SumProduct(grey_rect, conv1_weight_4, index_img + 32, 5, 5) +
                    SumProduct(grey_rect, conv1_weight_4, index_img + 64, 10, 5) +
                    SumProduct(grey_rect, conv1_weight_4, index_img + 96, 15, 5) +
                    SumProduct(grey_rect, conv1_weight_4, index_img + 128, 20, 5) + conv1_bias[4], 0);

                l2c5[index] = Math.Max(
                    SumProduct(grey_rect, conv1_weight_5, index_img, 0, 5) +
                    SumProduct(grey_rect, conv1_weight_5, index_img + 32, 5, 5) +
                    SumProduct(grey_rect, conv1_weight_5, index_img + 64, 10, 5) +
                    SumProduct(grey_rect, conv1_weight_5, index_img + 96, 15, 5) +
                    SumProduct(grey_rect, conv1_weight_5, index_img + 128, 20, 5) + conv1_bias[5], 0);
            }
        }


        // First maxpool (second layer to third layer) 
        double[] l3c0 = new double[14 * 14];
        double[] l3c1 = new double[14 * 14];
        double[] l3c2 = new double[14 * 14];
        double[] l3c3 = new double[14 * 14];
        double[] l3c4 = new double[14 * 14];
        double[] l3c5 = new double[14 * 14];
       ind_i = 0;
       ind_j = 0;
        for (int i = 0; i < 28; i += 2)
        {
            for (int j = 0; j < 28; j += 2)
            {
                l3c0[ind_i * 14 + ind_j] = Math.Max(l2c0[i * 28 + j], Math.Max(l2c0[i * 28 + j + 1], Math.Max(l2c0[(i + 1) * 28 + j], l2c0[(i + 1) * 28 + j + 1])));
                l3c1[ind_i * 14 + ind_j] = Math.Max(l2c1[i * 28 + j], Math.Max(l2c1[i * 28 + j + 1], Math.Max(l2c1[(i + 1) * 28 + j], l2c1[(i + 1) * 28 + j + 1])));
                l3c2[ind_i * 14 + ind_j] = Math.Max(l2c2[i * 28 + j], Math.Max(l2c2[i * 28 + j + 1], Math.Max(l2c2[(i + 1) * 28 + j], l2c2[(i + 1) * 28 + j + 1])));
                l3c3[ind_i * 14 + ind_j] = Math.Max(l2c3[i * 28 + j], Math.Max(l2c3[i * 28 + j + 1], Math.Max(l2c3[(i + 1) * 28 + j], l2c3[(i + 1) * 28 + j + 1])));
                l3c4[ind_i * 14 + ind_j] = Math.Max(l2c4[i * 28 + j], Math.Max(l2c4[i * 28 + j + 1], Math.Max(l2c4[(i + 1) * 28 + j], l2c4[(i + 1) * 28 + j + 1])));
                l3c5[ind_i * 14 + ind_j] = Math.Max(l2c5[i * 28 + j], Math.Max(l2c5[i * 28 + j + 1], Math.Max(l2c5[(i + 1) * 28 + j], l2c5[(i + 1) * 28 + j + 1])));
                ind_j++;
            }
            ind_i++;
            ind_j = 0;
        }


        //  Prediction (set pred based on the index of the maximum value)
        int maxIndex = 0;
                for (int i = 1; i < l3c0.Length; i++)
                {
                    if (l3c0[i] > l3c0[maxIndex])
                    {
                        maxIndex = i;
                    }
                }
        pred = maxIndex;

        for (int i = n1; i < n1 + 32 * 3; i++)
        {
            cols[i * targetTexture.width + m1] = Color.blue;
            cols[i * targetTexture.width + m1 + 32 * 3] = Color.blue;
        }
        for (int j = m1; j < m1 + 32 * 3; j++)
        {
            cols[n1 * targetTexture.width + j] = Color.blue;
            cols[(n1 + 32 * 3) * targetTexture.width + j] = Color.blue;
        }
        targetTexture.SetPixels32(cols);
        targetTexture.Apply();
        rndr.material.mainTexture = targetTexture;
        gameObject.GetComponent<Renderer>().enabled = true;
    }
    static double SumProduct(double[] array, double[] weights, int arrayStartIndex, int weightsStartIndex, int count)
    {
        double sum = 0;
        for (int k = 0; k < count; k++)
        {
            sum += array[arrayStartIndex + k] * weights[weightsStartIndex + k];
        }
        return sum;
    }
}