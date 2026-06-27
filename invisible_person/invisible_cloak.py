import cv2
import numpy as np
import time

# Step 1: Camera initialize karna (0 matlab default webcam)
cap = cv2.VideoCapture(0)

# Camera ko thoda time dete hain warm up hone aur brightness adjust karne ke liye
time.sleep(3)

background = 0

# Step 2: Background capture karna
# Note: Jab script run ho, toh shuru ke 3-4 seconds frame se bahar raho
# taaki yeh ek clean background capture kar sake.
for i in range(60):
    ret, background = cap.read()
    
# Image ko flip karte hain mirror effect ke liye
background = np.flip(background, axis=1) 

# Step 3: Real-time video processing
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    
    img = np.flip(img, axis=1)

    # BGR (Blue, Green, Red) image ko HSV (Hue, Saturation, Value) mein convert karna
    # Color detection ke liye HSV zyada accurate aur lighting-resistant hota hai
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 4: Red color ko detect karne ke liye HSV range define karna
    # Red color HSV spectrum mein start aur end dono jagah hota hai, isliye 2 masks banate hain
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Dono red masks ko combine karna
    mask = mask1 + mask2

    # Step 5: Mask ko thoda refine karna (noise hatana aur boundaries smooth karna)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Ek inverse mask banana (frame ka woh part jo red NAHI hai)
    mask_inv = cv2.bitwise_not(mask)

    # Step 6: Final Images extract karna
    # Sirf background ka woh hissa dikhao jahan red color detect hua hai (mask)
    res1 = cv2.bitwise_and(background, background, mask=mask)

    # Baki jagah original current frame dikhao jahan red color nahi hai (mask_inv)
    res2 = cv2.bitwise_and(img, img, mask=mask_inv)

    # Dono results ko jod do
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Output window dikhana
    cv2.imshow("Invisible Cloak by AADARSH", final_output)

    # 'q' dabane par exit karna
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()