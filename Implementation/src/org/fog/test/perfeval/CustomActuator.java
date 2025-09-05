package org.fog.test.perfeval;

import org.cloudbus.cloudsim.core.CloudSim;
import org.fog.entities.Actuator;
import org.fog.entities.Tuple;

public class CustomActuator extends Actuator {
    public CustomActuator(String name, int userId, String appId, String actuatorType) {
        super(name, userId, appId, actuatorType);
    }

    public void process(Tuple tuple) {
        if (tuple.getDirection() == Tuple.ACTUATOR && tuple.getTupleType().equals(this.getActuatorType())) {
            System.out.println("Actuator " + getName() + " received STOP tuple at time " + 
                               CloudSim.clock() + " (fault detected)");
        }
    }
}