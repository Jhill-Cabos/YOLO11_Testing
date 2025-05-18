elif "video" in file_type:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence, iou=iou_thresh)
        annotated_frame = np.squeeze(results[0].plot())

        class_ids = [int(result.cls) for result in results[0].boxes]
        label = classify_recklessness(class_ids)

        # Put the label at the top-left corner (say at coordinates (10,30))
        font_scale = 1.5 if label == "Reckless Driving" else 0.7
        color = (0, 0, 255) if label == "Reckless Driving" else (0, 255, 0)
        cv2.putText(annotated_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
