# Laser打靶

### 辨識方式一

形狀辨識 shape.py
    
1. 圖片要轉成二值圖像
2. 計算 中心座標:  周長: , 面積:  顏色:  形狀: 

色塊區域 selectcolr.py

1. 讀入圖片
2. 調整HVS分別找出“黑 灰 白 紅 橙 黃 綠 青 藍 紫”

模擬實際情況 VideoShape.py

1. 提供了60FPS影片來測試激光打在把上的影像,讀取出位置 a2.mp4
2. 顯示激光的位置,即共有幾個Frame有打到靶

### 辨識方式二（現行方式）

1. 取得Video: 320 x 240
2. Zoom out to 640 x 480
3. 前後Frame作比較 diff = cv.subtract(gray, res_old)
4. 過濾顏色,雜訊
5. 還原image size
6. 形狀辨識 - 計算 "中心座標:  周長: , 面積:  顏色:  形狀:"
7. 十字標記在原圖
