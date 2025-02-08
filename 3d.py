from folium.features import Template
import folium
import os

class Map3d(folium.Map):
    def __init__(self, location=None, width='100%', height='100%', left='0%',
                 top='0%', position='relative', tiles='OpenStreetMap', API_key=None,
                 max_zoom=18, min_zoom=1, zoom_start=10, attr=None, min_lat=-90,
                 max_lat=90, min_lon=-180, max_lon=180, detect_retina=False, crs='EPSG3857'):
        super(Map3d, self).__init__(
            location=location, width=width, height=height,
            left=left, top=top, position=position, tiles=tiles,
            API_key=API_key, max_zoom=max_zoom, min_zoom=min_zoom,
            zoom_start=zoom_start, attr=attr, min_lat=min_lat,
            max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,
            detect_retina=detect_retina, crs=crs
        )
        
        # Store markers & polygons
        self.markers = []
        self.polygons = []

        self._template = Template(u"""
        {% macro header(this, kwargs) %}
            <script src="https://www.webglearth.com/v2/api.js"></script>
            <style> #{{this.get_name()}} {
                position : {{this.position}};
                width : {{this.width[0]}}{{this.width[1]}};
                height: {{this.height[0]}}{{this.height[1]}};
                left: {{this.left[0]}}{{this.left[1]}};
                top: {{this.top[0]}}{{this.top[1]}};
                }
            </style>
        {% endmacro %}

        {% macro html(this, kwargs) %}
            <div class="folium-map" id="{{this.get_name()}}" ></div>
        {% endmacro %}

        {% macro script(this, kwargs) %}
            var {{this.get_name()}} = WE.map('{{this.get_name()}}', {
                center:[{{this.location[0]}}, {{this.location[1]}}],
                zoom: {{4}},
                layers: []
            });

            // Add base tile layer
            var baselayer = WE.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo({{this.get_name()}});

            // Add markers dynamically
            var markers = {{this.markers | tojson}};
            markers.forEach(function(m) {
                var marker = WE.marker([m.lat, m.lon]).addTo({{this.get_name()}});
                marker.bindPopup("<b>" + m.popup + "</b>");
            });

            // Add polygons dynamically
            var polygons = {{this.polygons | tojson}};
            polygons.forEach(function(p) {
                var latlngs = p.coords.map(c => [c[0], c[1]]);
                
                // Create polygon
                var polygon = WE.polygon(latlngs, {
                    strokeColor: p.color,  // Border color
                    fillColor: p.color,     // Fill color
                    fillOpacity: 0.5
                }).addTo({{this.get_name()}});

                // Calculate polygon center (average lat/lon)
                var centerLat = latlngs.reduce((sum, c) => sum + c[0], 0) / latlngs.length;
                var centerLon = latlngs.reduce((sum, c) => sum + c[1], 0) / latlngs.length;

                // Add invisible marker for the popup
                var popupMarker = WE.marker([centerLat, centerLon]).addTo({{this.get_name()}});
                popupMarker.bindPopup("<b>" + p.popup + "</b>");
            });

        {% endmacro %}
        """)

    def add_marker(self, lat, lon, popup_text="Marker"):
        """ Adds a marker dynamically to the map. """
        self.markers.append({"lat": lat, "lon": lon, "popup": popup_text})

    def add_polygon(self, coords, color="#FFFF00", popup_text="Polygon"):
        """ Adds a polygon dynamically to the map. """
        self.polygons.append({"coords": coords, "color": color, "popup": popup_text})


# Create a 3D map
m = Map3d(location=[0, 0], tiles=None, zoom_start=2)

# Add OpenStreetMap tiles using WebGL Earth
m.add_child(folium.TileLayer(tiles='OpenStreetMap'))

# Add markers dynamically
m.add_marker(40.7128, -74.0060, "New York")  
m.add_marker(51.5074, -0.1278, "London")  
m.add_marker(35.6895, 139.6917, "Tokyo")  

# Add polygons dynamically
m.add_polygon(
    coords=[[30, -10], [40, -20], [50, -10], [30, -10]],  
    color="#FFFF00",  # Yellow (HEX format)
    popup_text="Example Polygon"
)

# Save the map as an HTML file
m.save("3D_globe_dynamic.html")
