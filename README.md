# Dream4D: Lifting Camera-Controlled I2V towards Spatiotemporally Consistent 4D Generation

<div align="center">
    <a href="https://arxiv.org/abs/2508.07769"><img src="https://img.shields.io/static/v1?label=arXiv&message=2410.15957&color=b21d1a"></a>
    <a href="https://Wanderer7-sk.github.io/Dream4D.github.io"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=green"></a>
</div>

## Abstract:

The synthesis of spatiotemporally coherent 4D content presents fundamental challenges in computer vision, requiring simultaneous modeling of high-fidelity spatial representations and physically plausible temporal dynamics. Current approaches often struggle to maintain view consistency while handling complex scene dynamics, particularly in large-scale environments with multiple interacting elements. This work introduces **Dream4D**, a novel framework that bridges this gap through a *synergy of controllable video generation and neural 4D reconstruction*. Our approach seamlessly combines a two-stage architecture: it first predicts optimal camera trajectories from a single image using few-shot learning, then generates geometrically consistent multi-view sequences via a specialized pose-conditioned diffusion process, which are finally converted into a persistent 4D representation. This framework is the first to leverage both rich temporal priors from video diffusion models and geometric awareness of the reconstruction models, which significantly facilitates 4D generation and shows higher quality (e.g., mPSNR, mSSIM) over existing methods.

## 🚀 News and 📋 Todo List

- 🌱 25/8/5: We have uploaded part of the code, and the complete code will be open-sourced soon.

## 🎬 Demo Presentation

<table>
    <tr>
        <td align="center">
            Silver SUV cruises winding road through sunlit mountain valley.
        </td>
        <td align="center">
            A sleek red race car speeds down a marked track.
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/cdfb6399-236a-485c-945b-5ffcbe189712" alt="Silver SUV on mountain road" width="400">
        </td>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/4004b575-dd4a-4bed-8186-3bdfff675526" alt="Red race car on track" width="400">
        </td>
    </tr>
    <tr>
        <td align="center">
            Armored crocodile glides through shallow waters, its ridged back cutting ripples through mirrored wetlands.
        </td>
        <td align="center">
            The golden-maned lion strides purposefully down the sunlit savanna path, his amber eyes burning with regal intensity.
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/290a4e2f-cd21-4f7d-9434-e3cd458b4a57" alt="Crocodile in wetlands" width="400">
        </td>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/86bd48ef-2726-4c02-8624-693667e87379" alt="Lion on savanna" width="400">
        </td>
    </tr>
</table>
## 📊 Performance

### Performance Comparison of Different Methods

| Method          | mPSNR(dB) $\uparrow$ | mSSIM $\uparrow$ | mLPIPS $\downarrow$ |
| :-------------- | :------------------: | :--------------: | :-----------------: |
| **Ours**        |      **20.56**       |    **0.702**     |      **0.170**      |
| Megasam         |        17.625        |      0.601       |        0.207        |
| Shape-of-Motion |        16.72         |      0.630       |        0.450        |
| Cut3r           |        14.69         |      0.543       |        0.341        |
| Cami2v          |        14.08         |      0.449       |        0.334        |
| SeVA            |        12.67         |      0.495       |        0.579        |

### Model ablation study results

| Variant           | mPSNR(dB) $\uparrow$ | mSSIM $\uparrow$ | mLPIPS $\downarrow$ |
| :-------------------- | :----------: | :--------: | :---------------------------: |
| Dynamic Module + 4D Generator |   19.78 | 0.6686 |            0.1220       |
| Only Dynamic Module |   18.37   | 0.6361 |            0.1841       |
| $\delta$ | -1.41 | -0.0325 | +0.0621 |
| Static Module + 4D Generator | 13.35 | 0.6550 |            0.2996       |
| Only Static Module | 12.56 | 0.5779 |            0.3461     |
| $\delta$ | -0.79 | -0.0771 | +0.0465 |

## 🏗️ Static Module

The static module is specifically designed to build geometrically stable multi-view scenes, decoupling static scene elements from dynamic components to ensure view-consistent background reconstruction. 

## ⚡ Dynamic Module

The dynamic module is a specialized neural component that generates time-coherent video sequences by integrating camera trajectory control and dynamic content synthesis. It adopts some special mechanisms to maintain geometric consistency during viewpoint transitions, effectively decoupling foreground movements from static backgrounds and achieving physically reasonable synthesis of complex scene dynamics.

## 🔧 4D-Generator

Given a video or image sequence, it generates a dense 3D point cloud using self-supervised monocular depth and pose estimation. It serves as the core reconstruction engine, transforming temporally coherent video sequences into persistent neural 4D representations by integrating differentiable rendering with time-conditioned deformation fields.  It employs a transformer-based spatiotemporal feature field to jointly optimize geometry and motion dynamics, effectively mitigating shape drift and flickering artifacts through iterative alignment of surface details across frames.
### Features
- Supports video and image directory input
- Real-time 3D visualization via Viser
- Lightweight and easy to extend
- Clean, well-documented code base



**Our complete code will be coming soon**
