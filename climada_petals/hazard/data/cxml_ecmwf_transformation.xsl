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
    <xsl:apply-templates match="data[not(@type='analysis')]"/>
  </xsl:template>

  <xsl:template match="data">
    <xsl:apply-templates select="disturbance"/>
  </xsl:template>

  <xsl:template match="disturbance">
    <xsl:apply-templates select="fix[@source='model']">
      <xsl:with-param name="disturbance_no"><xsl:number /></xsl:with-param>
    </xsl:apply-templates>
  </xsl:template>

  <xsl:template match="fix">
    <xsl:param name="disturbance_no" />
    
    <xsl:variable name="latitude">
      <xsl:choose>
        <xsl:when test="latitude/@units = 'deg S'">
          <xsl:value-of select="concat('-', latitude)" />
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="latitude" />
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    
    <xsl:variable name="longitude">
      <xsl:choose>
        <xsl:when test="longitude/@units = 'deg W'">
          <xsl:value-of select="concat('-', longitude)" />
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="longitude" />
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>

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
      validTime, ',',
      $latitude, ',',
      $longitude, ',',
      cycloneData/minimumPressure/pressure, ',',
      cycloneData/lastClosedIsobar/pressure, ',',
      cycloneData/maximumWind/speed, ',',
      cycloneData/maximumWind/radius,
      $newline
      )" />
  </xsl:template>
</xsl:stylesheet>
