simple UdpDelayRelay
{
    parameters:
        int listenPort;
        string destAddress = default("127.0.0.1");
        int destPort;
        double fixedDelay @unit(s) = default(0.05s);
        double jitter @unit(s) = default(0.01s);
        double dropProbability = default(0);
        double pollInterval @unit(s) = default(0.01s);
        @display("i=block/wrxtx");
}

network HuskyToUavRelay
{
    @display("bgb=220,120");
    submodules:
        relay: UdpDelayRelay {
            @display("p=110,60");
        }
}
