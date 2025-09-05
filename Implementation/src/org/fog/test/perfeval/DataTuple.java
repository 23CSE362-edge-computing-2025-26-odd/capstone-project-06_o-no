package org.fog.test.perfeval;

import org.cloudbus.cloudsim.UtilizationModelFull;
import org.fog.entities.Tuple;

import java.util.HashMap;
import java.util.Map;

public class DataTuple extends Tuple {
    private Map<String, Object> payload = new HashMap<>();
    private long payloadSize;
    private String destModuleName;

    public DataTuple(String appId, int cloudletId, int direction, long cloudletLength, int pesNumber,
                     long cloudletFileSize, long cloudletOutputSize,
                     UtilizationModelFull cpuUtilizationModel, UtilizationModelFull ramUtilizationModel,
                     UtilizationModelFull bwUtilizationModel) {
        super(appId, cloudletId, direction, cloudletLength, pesNumber, cloudletFileSize, cloudletOutputSize,
              cpuUtilizationModel, ramUtilizationModel, bwUtilizationModel);
        this.payloadSize = cloudletFileSize;
    }

    public Map<String, Object> getPayload() {
        return payload;
    }

    public void setPayload(Map<String, Object> payload) {
        this.payload = payload;
        this.payloadSize = payload.toString().getBytes().length;
    }

    public long getPayloadSize() {
        return payloadSize;
    }

    public void setDestModuleName(String destModuleName) {
        this.destModuleName = destModuleName;
    }

    public String getDestModuleName() {
        return destModuleName;
    }
}