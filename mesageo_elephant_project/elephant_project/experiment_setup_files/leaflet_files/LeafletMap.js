var MapModule = function (view, zoom, map_width, map_height) {
  // Create the map tag:
  var map_tag = "<div style='width:" + map_width + "px; height:" + map_height + "px;border:1px dotted' id='mapid'></div>"
  // Append it to body:
  var div = $(map_tag)[0]
  $('#elements').append(div)

  // Create Leaflet map and Agent layers
  var Lmap = L.map('mapid').setView(view, zoom)
  var AgentLayer = L.geoJSON().addTo(Lmap)
  var AgentFitness = L.geoJSON().addTo(Lmap)
  var AgentAggression = L.geoJSON().addTo(Lmap)
 
  // create the OSM tile layer with correct attribution
  var osmUrl = 'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
  var osmAttrib = 'Map data © <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
  var osm = new L.TileLayer(osmUrl, { minZoom: 0, maxZoom: 18, attribution: osmAttrib })
  Lmap.addLayer(osm)

  var imageUrl = 'https://i.ibb.co/YQX0XyK/LULC.png',
  imageBounds = [[9.177469950957937, 76.84843237643445], [9.471468123855534, 77.14636762356552]];
  L.imageOverlay(imageUrl, imageBounds,  {opacity: 0.75} ).addTo(Lmap);

  // define rectangle geographical bounds
  var bounds = [[9.177469950957937, 76.84843237643445], [9.471468123855534, 77.14636762356552]];

  // create an orange rectangle
  L.rectangle(bounds, {color: "#ff7800", weight: 5}).addTo(Lmap);

  var ElephantIcon = L.icon({
  iconUrl: 'https://i.ibb.co/PNhQP8T/elephant-1.png',
  iconSize:     [25, 25], // size of the icon
  });


  var HumanIcon = L.icon({
  iconUrl: 'https://i.ibb.co/khGGfnX/png-hd-of-stick-figures-transparent-hd-of-stick-figures-images-363400.png',
  iconSize:     [25, 25], // size of the icon
  });

  this.render = function (data) {
    AgentLayer.remove()
    AgentFitness.remove()
    AgentAggression.remove()

    AgentLayer = L.geoJSON(data, {
      onEachFeature: PopUpProperties,
      style: function (feature) {
        return { color: feature.properties.color };
      },
      pointToLayer: function (feature, latlang) {

	if (feature.properties.category == "Elephant") {
	return L.marker(latlang , {icon: ElephantIcon});}

	else if (feature.properties.category == "Human"){
	return L.marker(latlang , {icon: HumanIcon});}

	else {
	return L.circle(latlang, { radius: feature.properties.radius, color: feature.properties.color });}

      }
    }).addTo(Lmap)

    AgentFitness = L.geoJSON(data, {
       pointToLayer: function (feature, latlang) {

  if (feature.properties.category == "Elephant"){
          return  L.marker(latlang, {
            icon: L.divIcon({
            html: feature.properties.fitness,
            className: 'text-below-marker_fitness',})
            })
    }  }
   }) .addTo(Lmap);

    AgentAggression = L.geoJSON(data, {
       pointToLayer: function (feature, latlang) {
  if (feature.properties.category == "Elephant"){
          return  L.marker(latlang, {
            icon: L.divIcon({
            html: feature.properties.aggression,
            className: 'text-below-marker_aggression',})
            })
    }  }
   }).addTo(Lmap);

  }

  this.reset = function () {
    AgentLayer.remove()
    AgentFitness.remove()
    AgentAggression.remove()
  }
}


function PopUpProperties(feature, layer) {
  var popupContent = '<table>'
  if (feature.properties) {
    for (var p in feature.properties) {
      popupContent += '<tr><td>' + p + '</td><td>' + feature.properties[p] + '</td></tr>'
    }
  }
  popupContent += '</table>'
  layer.bindPopup(popupContent)
}