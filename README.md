# **InterIIT Bosch's Age And Gender Detection**

## PROBLEM STATEMENT

## DESCRIPTION

The scenes obtained from a surveillance video are usually with low resolution. Most of the scenes captured by a static camera are with minimal change of background. Objects in outdoor surveillance are often detected in far-fields. Most existing digital video surveillance systems rely on human observers for detecting specific activities in a real-time video scene. However, there are limitations in the human capability to monitor simultaneous events in surveillance displays. Hence, human motion analysis in automated video surveillance has become one of the most active and attractive research topics in the area of computer vision and pattern recognition.

## PAIN POINT

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: How do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution.

## PROBLEM STATEMENT

Build a solution to estimate the gender and age of people from a surveillance video feed (like mall, retail store, hospital etc.). Consider low resolution cameras as well as cameras put at a height for surveillance. Hint: Develop a pipeline to apply super resolution algorithms for people detected and then apply gender/age estimation algorithms. Problem approach:
1. Object Detection: SRGAN algorithms.
2.Classification and estimation: Deep learning and neural networks
( e.g. : this link )
Link for full paper: link
Domains: Super Resolution GAN, DL, Image Processing

## Steps to run

1. Clone this repo locally by using fllowing command in your terminal.

```
git clone git@github.com:yashcode00/InterIIT.git
```
2. Then, install all the dependicies by using the follwing commands on same terminal.

```
cd InterIIT
pip install -r requirements.txt
```

3. Then finally run following code and you will be good to go :)

```
python main.py
```

Thanks & Regards
Team
