import cv2
import numpy as np
def k_max_values_and_indices(scores, k,r=100,threshold=0.0):
    # Flatten the array and get the indices of the top-k values

    preds_list = []
    maxvals_list = []

    for _ in range(k):
        idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)

        maxval = scores[idx]
        if maxval<threshold:
            break
        maxvals_list.append(float(maxval))
        preds_list.append(idx)
        # Clear the square region around the point
        x, y = idx[0], idx[1]
        xmin, ymin = max(0, x - r // 2), max(0, y - r // 2)
        xmax, ymax = min(scores.shape[0], x + r // 2), min(scores.shape[1], y + r // 2)
        scores[ xmin:xmax,ymin:ymax] = -9
    maxvals_list=np.array(maxvals_list,dtype=np.float16)
    preds_list=np.array(preds_list,dtype=np.float16)
    return maxvals_list, preds_list

def visual_points(image_path,pred_mask, save_path,points_number=6):
    img = cv2.imread(image_path)
    w_r,h_r=img.shape[0]/pred_mask.shape[0],img.shape[1]/pred_mask.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    maxval,pred_point=k_max_values_and_indices(pred_mask,points_number)
    # Draw landmarks on the image
    # print(pred_point)
    pred_point[:,0]*=w_r
    pred_point[:,1]*=h_r
    cnt=1
    for pred in pred_point:
        x, y = pred
        # x,y=x*w_r,y*h_r
        cv2.circle(img, (int(y), int(x)), 8, (255, 0, 0), -1)
        cv2.putText(img, f"{cnt}", (int(y), int(x)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cnt+=1
    # Save the image with landmarks
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
