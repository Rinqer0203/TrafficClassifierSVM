from enum import IntEnum


class TrafficAttr(IntEnum):
    DURATION = 0
    SERVICE = 1
    SOURCE_BYTES = 2
    DESTINATION_BYTES = 3
    COUNT = 4
    SAME_SRV_RATE = 5
    SERROR_RATE = 6
    SRV_SERROR_RATE = 7
    DST_HOST_COUNT = 8
    DST_HOST_SRV_COUNT = 9
    DST_HOST_SAME_SRC_PORT_RATE = 10
    DST_HOST_SERROR_RATE = 11
    DST_HOST_SRV_SERROR_RATE = 12
    FLAG = 13
    IDS_DETECTION = 14  # 0: Normal, 1: Alert
    MALWARE_DETECTION = 15  # 0: Normal, 1: Name
    ASHULA_DETECTION = 16  # 0: Normal, 1: Name
    LABEL = 17  # 1: Normal, -1: Known Attack, -2: Unknown Attack
    SOURCE_IP_ADDRESS = 18
    SOURCE_PORT_NUMBER = 19
    DESTINATION_IP_ADDRESS = 20
    DESTINATION_PORT_NUMBER = 21
    START_TIME = 22
    PROTOCOL = 23

    def get_attribute_name_list() -> list:
        return list(TrafficAttr.__members__.keys())
