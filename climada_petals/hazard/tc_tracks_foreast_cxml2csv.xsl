<?xml version="1.0" encoding="UTF-8"?>
<!-- 
  Purpose: Extract operational and ensemble cyclone xml (aka cxml) track 
    forecast data into CSV
  Caveat: Omitted double ticks " to demarcate CSV fields; running on faith/the
    assumption that no commas appear in any fields.
  Author: Jan Hartman
-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

  <xsl:output method="text"/>
  <xsl:strip-space elements="*"/>
  <xsl:variable name="newline">
    <!-- linefeed character code -->
    <xsl:text>&#xa;</xsl:text> 
  </xsl:variable>

  <xsl:template match="header"></xsl:template>

  <xsl:template match="cxml">
    <xsl:value-of select="concat(
      'disturbance_no,',
      'baseTime,',
      'origin,',
      'type,',
      'member,',
      'perturb,',
      'id,',
      'cycloneName,',
      'cycloneNumber,',
      'basin,',
      'hour,',
      'validTime,',
      'latitude,',
      'longitude,',
      'minimumPressure,',
      'lastClosedIsobar,',
      'maximumWind,',
      'maximumWindRadius',
      $newline
      )"
    />
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="data[not(@type='analysis')]">
    <xsl:apply-templates select="disturbance"/>
  </xsl:template>

  <xsl:template match="disturbance">
    <xsl:apply-templates select="fix[@source='model']">
      <xsl:with-param name="disturbance_no"><xsl:number /></xsl:with-param>
    </xsl:apply-templates>
  </xsl:template>

  <xsl:template match="fix[@source='model']">
    <xsl:param name="disturbance_no" />
    <xsl:value-of select="concat(
      $disturbance_no, ',',
      ancestor::cxml/header/baseTime, ',',
      ancestor::data/@origin, ',',
      ancestor::data/@type, ',',
      ancestor::data/@member, ',',
      ancestor::data/@perturb, ',',
      ancestor::disturbance/@ID, ',',
      ancestor::disturbance/cycloneName, ',',
      ancestor::disturbance/cycloneNumber, ',',
      ancestor::disturbance/basin, ',',
      @hour, ',',
      descendant::validTime, ',',
      descendant::latitude, ',',
      descendant::longitude, ',',
      descendant::cycloneData/minimumPressure/pressure, ',',
      descendant::cycloneData/lastClosedIsobar/pressure, ',',
      descendant::cycloneData/maximumWind/speed, ',',
      descendant::cycloneData/maximumWind/radius,
      $newline
      )" />
  </xsl:template>
</xsl:stylesheet>
