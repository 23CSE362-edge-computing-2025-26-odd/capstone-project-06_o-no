package org.fog.test.perfeval;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.ResCloudlet;
import org.fog.entities.Tuple;
import org.fog.scheduler.TupleScheduler;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.logging.Logger;

/**
 * Custom Scheduler: Priority-based tuple scheduling (e.g., SENSOR tuples first).
 */
public class CustomTupleScheduler extends TupleScheduler {
    private static final Logger LOGGER = Logger.getLogger(CustomTupleScheduler.class.getName());
    private PriorityQueue<Tuple> tupleQueue;
    private final double mips; // Store MIPS explicitly

    public CustomTupleScheduler(double mips, int numberOfPes) {
        super(mips, numberOfPes);
        this.mips = mips;
        tupleQueue = new PriorityQueue<>(Comparator.comparingInt(this::getPriority));
    }

    /**
     * Submits a tuple (cloudlet) for execution, prioritizing based on tuple type.
     * @param cl the cloudlet (tuple) to submit
     * @return expected time to complete the cloudlet
     */
    @Override
    public double cloudletSubmit(Cloudlet cl) {
        if (cl instanceof Tuple) {
            Tuple tuple = (Tuple) cl;
            tupleQueue.offer(tuple);
            LOGGER.info("Added tuple " + tuple.getTupleType() + " to queue with priority " + getPriority(tuple) + " at time " + org.cloudbus.cloudsim.core.CloudSim.clock());

            while (!tupleQueue.isEmpty()) {
                Tuple nextTuple = tupleQueue.peek();
                // Check if there are enough resources to process the tuple
                double requestedMips = getCurrentRequestedMips() != null ?
                    getCurrentRequestedMips().stream().mapToDouble(Double::doubleValue).sum() : 0;
                if (requestedMips < mips) {
                    tupleQueue.poll();
                    double completionTime = super.cloudletSubmit(nextTuple);
                    LOGGER.info("Submitted tuple " + nextTuple.getTupleType() + " for execution at time " + org.cloudbus.cloudsim.core.CloudSim.clock());
                    return completionTime;
                } else {
                    break; // Insufficient MIPS, keep tuple in queue
                }
            }
            return -1; // Tuple queued but not submitted due to insufficient resources
        }
        return super.cloudletSubmit(cl); // Handle non-tuple cloudlets
    }

    /**
     * Handles tuple completion.
     * @param rcl the ResCloudlet (tuple) that finished
     */
    @Override
    public void cloudletFinish(ResCloudlet rcl) {
        if (rcl.getCloudlet() instanceof Tuple) {
            Tuple tuple = (Tuple) rcl.getCloudlet();
            LOGGER.info("Finished tuple " + tuple.getTupleType() + " at time " + org.cloudbus.cloudsim.core.CloudSim.clock());
        }
        super.cloudletFinish(rcl);
    }

    /**
     * Determines priority based on tuple type.
     * @param tuple the tuple to prioritize
     * @return priority value (lower is higher priority)
     */
    private int getPriority(Tuple tuple) {
        switch (tuple.getTupleType()) {
            case "SENSOR":
                return 1; // Highest priority
            case "PROCESSED_DATA":
                return 2;
            default:
                return 3; // Lowest priority
        }
    }
}