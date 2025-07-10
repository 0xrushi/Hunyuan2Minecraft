# 🗼 Hunyuan2Minecraft: Build Beautiful Structures in Minecraft Using Images + AI

Turn stunning real-world or AI-generated images into Minecraft builds — powered by [Hunyuan 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/) and voxel conversion pipelines.

## 🎥 Demo (click the image)

[![Watch the Demo Video](https://github.com/user-attachments/assets/51702fa5-feee-41c5-9100-948e25c3bff5)](https://www.youtube.com/watch?v=d4WiroXOokU)
> Click to watch the Eiffel Tower built in Minecraft using an image and Hunyuan's imagination.

---

## 💡 Why?

Minecraft agents are getting better at chopping trees and mining but when it comes to building **realistic**, **beautiful**, and **creative** structures, they fail.

Inspired by projects like [Claude building the Eiffel Tower (poorly 😬)](https://www.reddit.com/r/mcp/comments/1jgicku/claudes_building_the_eiffel_tower_in_realtime/), this project bridges the gap between **vision models** and **blocky reality**.
![Screenshot 2025-07-08 044704](https://github.com/user-attachments/assets/be7f1b4b-6659-4d13-a753-7cdf7ea715c1)


---

## 🧠 How It Works

1. **Image Input**  
   Provide a real-world or AI-generated image of a structure.

2. **Hunyuan 2.1 Vision Model**  
   We extract structural and spatial data using Hunyuan.

3. **Voxelization**  
   Convert image → 3D voxel matrix (supports STL/OBJ pipelines or direct voxel inference).

4. **Minecraft Block Mapping**  
   Map voxel materials to Minecraft blocks intelligently.


-- 

## ToDO 

[ ] Implement cluster detection unsupervised algorithm to automatically detect colors for generated textures

[ ] Develop an algorithm to convert the color hues to minecraft blocks like orange = pumpkin block
