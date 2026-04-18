#include <omnetpp.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

using namespace omnetpp;

namespace {
class RelayPacketMessage : public cMessage {
  public:
    std::string payload;

    RelayPacketMessage(const char *name, std::string data)
        : cMessage(name), payload(std::move(data)) {}
};
}

class UdpDelayRelay : public cSimpleModule
{
  private:
    int sockfd = -1;
    sockaddr_in destAddr {};
    cMessage *pollMsg = nullptr;
    simtime_t fixedDelay;
    simtime_t jitter;
    double dropProbability = 0.0;
    simtime_t pollInterval;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;

    void pollSocket();
    void sendPayload(const std::string& payload);
};

Define_Module(UdpDelayRelay);

void UdpDelayRelay::initialize()
{
    fixedDelay = par("fixedDelay");
    jitter = par("jitter");
    dropProbability = par("dropProbability").doubleValue();
    pollInterval = par("pollInterval");

    sockfd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
        throw cRuntimeError("socket() failed: %s", strerror(errno));

    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags < 0 || fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0)
        throw cRuntimeError("fcntl(O_NONBLOCK) failed: %s", strerror(errno));

    int reuse = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
        throw cRuntimeError("setsockopt(SO_REUSEADDR) failed: %s", strerror(errno));

    sockaddr_in localAddr {};
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = htons(par("listenPort").intValue());
    if (::bind(sockfd, reinterpret_cast<sockaddr *>(&localAddr), sizeof(localAddr)) < 0)
        throw cRuntimeError("bind() failed on port %d: %s", par("listenPort").intValue(), strerror(errno));

    destAddr.sin_family = AF_INET;
    destAddr.sin_port = htons(par("destPort").intValue());
    std::string destAddress = par("destAddress").stdstringValue();
    if (inet_aton(destAddress.c_str(), &destAddr.sin_addr) == 0)
        throw cRuntimeError("Invalid destination address: %s", destAddress.c_str());

    pollMsg = new cMessage("pollSocket");
    scheduleAt(simTime() + pollInterval, pollMsg);
}

void UdpDelayRelay::handleMessage(cMessage *msg)
{
    if (msg == pollMsg) {
        pollSocket();
        scheduleAt(simTime() + pollInterval, pollMsg);
        return;
    }

    auto *relayMsg = check_and_cast<RelayPacketMessage *>(msg);
    sendPayload(relayMsg->payload);
    delete relayMsg;
}

void UdpDelayRelay::pollSocket()
{
    char buffer[2048];
    while (true) {
        sockaddr_in sourceAddr {};
        socklen_t sourceLen = sizeof(sourceAddr);
        ssize_t received = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                                    reinterpret_cast<sockaddr *>(&sourceAddr), &sourceLen);
        if (received < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN)
                break;
            throw cRuntimeError("recvfrom() failed: %s", strerror(errno));
        }

        if (uniform(0.0, 1.0) < dropProbability)
            continue;

        buffer[received] = '\0';
        simtime_t delay = fixedDelay;
        if (jitter > SIMTIME_ZERO)
            delay += uniform(-jitter.dbl(), jitter.dbl());
        if (delay < SIMTIME_ZERO)
            delay = SIMTIME_ZERO;

        scheduleAt(simTime() + delay, new RelayPacketMessage("relayPayload", std::string(buffer, received)));
    }
}

void UdpDelayRelay::sendPayload(const std::string& payload)
{
    ssize_t sent = sendto(sockfd, payload.data(), payload.size(), 0,
                          reinterpret_cast<sockaddr *>(&destAddr), sizeof(destAddr));
    if (sent < 0)
        EV_WARN << "sendto() failed: " << strerror(errno) << "\n";
}

void UdpDelayRelay::finish()
{
    cancelAndDelete(pollMsg);
    pollMsg = nullptr;
    if (sockfd >= 0)
        ::close(sockfd);
}
