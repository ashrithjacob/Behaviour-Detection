def transform_fn(model, request_body, content_type, accept_type):
    dmatrix = xgb_encoders.libsvm_to_dmatrix(request_body)
    prediction = model.predict(dmatrix)
    feature_contribs = model.predict(
        dmatrix, pred_contribs=True, validate_features=False
    )
    output = np.hstack((prediction[:, np.newaxis], feature_contribs))
    return ",".join(str(x) for x in predictions[0])
