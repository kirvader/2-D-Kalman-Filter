import cv2
from BasicBallDetector import detect
from KalmanFilter import KalmanFilter


def main():
    # Create opencv video capture object
    VideoCap = cv2.VideoCapture('video/randomball.avi')

    # Variable used to control the speed of reading the video
    ControlSpeedVar = 100

    HiSpeed = 100

    # Create KalmanFilter object KF
    # KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

    debugMode = 1

    while (True):
        # Read frame
        ret, frame = VideoCap.read()

        # Detect object
        centers = detect(frame, debugMode)

        # If centroids are detected then track them
        if (len(centers) > 0):
            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)

            # Predict
            (x, y) = KF.predict()
            x, y = int(x), int(y)
            # Draw a rectangle as the predicted object position
            cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), 2)

            # Update
            (x1, y1) = KF.update(centers[0])
            x1, y1 = int(x1), int(y1)

            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (x1 - 15, y1 - 15), (x1 + 15, y1 + 15), (0, 0, 255), 2)

            cv2.putText(frame, "Estimated Position", (x1 + 15, y1 + 10), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (x + 15, y), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (int(centers[0][0]) + 15, int(centers[0][1]) - 15), 0, 0.5,
                        (0, 191, 255), 2)

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed - ControlSpeedVar + 1)


if __name__ == "__main__":
    # execute main
    main()
