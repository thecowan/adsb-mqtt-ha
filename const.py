"""General adsb_info constants."""
from homeassistant.const import Platform

DOMAIN = "adsb_info"
PLATFORMS = [Platform.SENSOR]
CONF_ADSB_SENSOR = "adsb_sensor"
CONF_ADSB_JSON_ATTRIBUTE = "adsb_json_attribute"
CONF_POLL = "poll"

DEFAULT_NAME = "ADSB Info"
UPDATE_LISTENER = "update_listener"
