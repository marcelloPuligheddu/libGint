int locate_patch_low_R( const double X1, const double X2 ){

   if (X1 <= 0.500000000000000000E+00) { 
      if (X2 <= 0.500000000000000000E+00) { 
         if (X2 <= 0.250000000000000000E+00) { 
            if (X2 <= 0.125000000000000000E+00) { 
               if (X1 <= 0.250000000000000000E+00) { 
                  if (X2 <= 0.625000000000000000E-01) { 
                     if (X1 <= 0.125000000000000000E+00) { 
                        if (X2 <= 0.312500000000000000E-01) { 
                           if (X1 <= 0.625000000000000000E-01) { 
                              if (X2 <= 0.156250000000000000E-01) { 
                                 if (X1 <= 0.312500000000000000E-01) { 
                                    return 0;
                                 } else {
                                    return 1;
                                 }
                              } else {
                                 if (X1 <= 0.312500000000000000E-01) { 
                                    return 2;
                                 } else {
                                    return 3;
                                 }
                              }
                           } else {
                              if (X2 <= 0.156250000000000000E-01) { 
                                 return 4;
                              } else {
                                 return 5;
                              }
                           }
                        } else {
                           if (X1 <= 0.625000000000000000E-01) { 
                              if (X2 <= 0.468750000000000000E-01) { 
                                 if (X1 <= 0.312500000000000000E-01) { 
                                    return 6;
                                 } else {
                                    return 7;
                                 }
                              } else {
                                 if (X1 <= 0.312500000000000000E-01) { 
                                    return 8;
                                 } else {
                                    return 9;
                                 }
                              }
                           } else {
                              if (X2 <= 0.468750000000000000E-01) { 
                                 return 10;
                              } else {
                                 return 11;
                              }
                           }
                        }
                     } else {
                        if (X2 <= 0.312500000000000000E-01) { 
                           if (X1 <= 0.187500000000000000E+00) { 
                              return 12;
                           } else {
                              return 13;
                           }
                        } else {
                           if (X1 <= 0.187500000000000000E+00) { 
                              return 14;
                           } else {
                              return 15;
                           }
                        }
                     }
                  } else {
                     if (X1 <= 0.125000000000000000E+00) { 
                        if (X2 <= 0.937500000000000000E-01) { 
                           if (X1 <= 0.625000000000000000E-01) { 
                              if (X2 <= 0.781250000000000000E-01) { 
                                 if (X1 <= 0.312500000000000000E-01) { 
                                    return 16;
                                 } else {
                                    return 17;
                                 }
                              } else {
                                 if (X1 <= 0.312500000000000000E-01) { 
                                    return 18;
                                 } else {
                                    return 19;
                                 }
                              }
                           } else {
                              if (X2 <= 0.781250000000000000E-01) { 
                                 return 20;
                              } else {
                                 return 21;
                              }
                           }
                        } else {
                           if (X1 <= 0.625000000000000000E-01) { 
                              if (X2 <= 0.109375000000000000E+00) { 
                                 if (X1 <= 0.312500000000000000E-01) { 
                                    return 22;
                                 } else {
                                    return 23;
                                 }
                              } else {
                                 if (X1 <= 0.312500000000000000E-01) { 
                                    return 24;
                                 } else {
                                    return 25;
                                 }
                              }
                           } else {
                              if (X2 <= 0.109375000000000000E+00) { 
                                 return 26;
                              } else {
                                 return 27;
                              }
                           }
                        }
                     } else {
                        if (X1 <= 0.187500000000000000E+00) { 
                           if (X2 <= 0.937500000000000000E-01) { 
                              return 28;
                           } else {
                              return 29;
                           }
                        } else {
                           return 30;
                        }
                     }
                  }
               } else {
                  if (X1 <= 0.375000000000000000E+00) { 
                     return 31;
                  } else {
                     return 32;
                  }
               }
            } else {
               if (X1 <= 0.250000000000000000E+00) { 
                  if (X2 <= 0.187500000000000000E+00) { 
                     if (X1 <= 0.125000000000000000E+00) { 
                        if (X2 <= 0.156250000000000000E+00) { 
                           if (X1 <= 0.625000000000000000E-01) { 
                              if (X1 <= 0.312500000000000000E-01) { 
                                 if (X2 <= 0.140625000000000000E+00) { 
                                    return 33;
                                 } else {
                                    return 34;
                                 }
                              } else {
                                 return 35;
                              }
                           } else {
                              if (X1 <= 0.937500000000000000E-01) { 
                                 return 36;
                              } else {
                                 return 37;
                              }
                           }
                        } else {
                           if (X1 <= 0.625000000000000000E-01) { 
                              if (X1 <= 0.312500000000000000E-01) { 
                                 if (X2 <= 0.171875000000000000E+00) { 
                                    return 38;
                                 } else {
                                    return 39;
                                 }
                              } else {
                                 return 40;
                              }
                           } else {
                              return 41;
                           }
                        }
                     } else {
                        if (X1 <= 0.187500000000000000E+00) { 
                           return 42;
                        } else {
                           return 43;
                        }
                     }
                  } else {
                     if (X1 <= 0.125000000000000000E+00) { 
                        if (X1 <= 0.625000000000000000E-01) { 
                           if (X2 <= 0.218750000000000000E+00) { 
                              if (X1 <= 0.312500000000000000E-01) { 
                                 if (X2 <= 0.203125000000000000E+00) { 
                                    return 44;
                                 } else {
                                    return 45;
                                 }
                              } else {
                                 return 46;
                              }
                           } else {
                              if (X1 <= 0.312500000000000000E-01) { 
                                 if (X2 <= 0.234375000000000000E+00) { 
                                    return 47;
                                 } else {
                                    return 48;
                                 }
                              } else {
                                 return 49;
                              }
                           }
                        } else {
                           if (X2 <= 0.218750000000000000E+00) { 
                              return 50;
                           } else {
                              return 51;
                           }
                        }
                     } else {
                        if (X1 <= 0.187500000000000000E+00) { 
                           return 52;
                        } else {
                           return 53;
                        }
                     }
                  }
               } else {
                  if (X1 <= 0.375000000000000000E+00) { 
                     if (X1 <= 0.312500000000000000E+00) { 
                        return 54;
                     } else {
                        return 55;
                     }
                  } else {
                     return 56;
                  }
               }
            }
         } else {
            if (X1 <= 0.250000000000000000E+00) { 
               if (X1 <= 0.125000000000000000E+00) { 
                  if (X1 <= 0.625000000000000000E-01) { 
                     if (X2 <= 0.375000000000000000E+00) { 
                        if (X2 <= 0.312500000000000000E+00) { 
                           if (X1 <= 0.312500000000000000E-01) { 
                              if (X2 <= 0.281250000000000000E+00) { 
                                 return 57;
                              } else {
                                 return 58;
                              }
                           } else {
                              if (X2 <= 0.281250000000000000E+00) { 
                                 return 59;
                              } else {
                                 return 60;
                              }
                           }
                        } else {
                           if (X1 <= 0.312500000000000000E-01) { 
                              if (X2 <= 0.343750000000000000E+00) { 
                                 return 61;
                              } else {
                                 return 62;
                              }
                           } else {
                              return 63;
                           }
                        }
                     } else {
                        if (X1 <= 0.312500000000000000E-01) { 
                           if (X2 <= 0.437500000000000000E+00) { 
                              if (X1 <= 0.156250000000000000E-01) { 
                                 return 64;
                              } else {
                                 return 65;
                              }
                           } else {
                              if (X1 <= 0.156250000000000000E-01) { 
                                 return 66;
                              } else {
                                 return 67;
                              }
                           }
                        } else {
                           if (X2 <= 0.437500000000000000E+00) { 
                              return 68;
                           } else {
                              return 69;
                           }
                        }
                     }
                  } else {
                     if (X2 <= 0.375000000000000000E+00) { 
                        if (X2 <= 0.312500000000000000E+00) { 
                           if (X1 <= 0.937500000000000000E-01) { 
                              return 70;
                           } else {
                              return 71;
                           }
                        } else {
                           if (X1 <= 0.937500000000000000E-01) { 
                              return 72;
                           } else {
                              return 73;
                           }
                        }
                     } else {
                        if (X1 <= 0.937500000000000000E-01) { 
                           if (X2 <= 0.437500000000000000E+00) { 
                              return 74;
                           } else {
                              return 75;
                           }
                        } else {
                           if (X2 <= 0.437500000000000000E+00) { 
                              return 76;
                           } else {
                              return 77;
                           }
                        }
                     }
                  }
               } else {
                  if (X2 <= 0.375000000000000000E+00) { 
                     if (X1 <= 0.187500000000000000E+00) { 
                        if (X2 <= 0.312500000000000000E+00) { 
                           return 78;
                        } else {
                           if (X1 <= 0.156250000000000000E+00) { 
                              return 79;
                           } else {
                              return 80;
                           }
                        }
                     } else {
                        if (X2 <= 0.312500000000000000E+00) { 
                           return 81;
                        } else {
                           return 82;
                        }
                     }
                  } else {
                     if (X1 <= 0.187500000000000000E+00) { 
                        if (X2 <= 0.437500000000000000E+00) { 
                           return 83;
                        } else {
                           if (X1 <= 0.156250000000000000E+00) { 
                              return 84;
                           } else {
                              return 85;
                           }
                        }
                     } else {
                        if (X1 <= 0.218750000000000000E+00) { 
                           return 86;
                        } else {
                           return 87;
                        }
                     }
                  }
               }
            } else {
               if (X1 <= 0.375000000000000000E+00) { 
                  if (X2 <= 0.375000000000000000E+00) { 
                     if (X1 <= 0.312500000000000000E+00) { 
                        return 88;
                     } else {
                        return 89;
                     }
                  } else {
                     if (X1 <= 0.312500000000000000E+00) { 
                        if (X1 <= 0.281250000000000000E+00) { 
                           return 90;
                        } else {
                           return 91;
                        }
                     } else {
                        return 92;
                     }
                  }
               } else {
                  if (X1 <= 0.437500000000000000E+00) { 
                     return 93;
                  } else {
                     return 94;
                  }
               }
            }
         }
      } else {
         if (X1 <= 0.250000000000000000E+00) { 
            if (X1 <= 0.125000000000000000E+00) { 
               if (X1 <= 0.625000000000000000E-01) { 
                  if (X1 <= 0.312500000000000000E-01) { 
                     if (X1 <= 0.156250000000000000E-01) { 
                        if (X1 <= 0.781250000000000000E-02) { 
                           if (X2 <= 0.750000000000000000E+00) { 
                              return 95;
                           } else {
                              return 96;
                           }
                        } else {
                           if (X2 <= 0.750000000000000000E+00) { 
                              return 97;
                           } else {
                              return 98;
                           }
                        }
                     } else {
                        if (X2 <= 0.750000000000000000E+00) { 
                           return 99;
                        } else {
                           return 100;
                        }
                     }
                  } else {
                     if (X2 <= 0.750000000000000000E+00) { 
                        if (X2 <= 0.625000000000000000E+00) { 
                           return 101;
                        } else {
                           return 102;
                        }
                     } else {
                        if (X1 <= 0.468750000000000000E-01) { 
                           return 103;
                        } else {
                           return 104;
                        }
                     }
                  }
               } else {
                  if (X2 <= 0.750000000000000000E+00) { 
                     if (X2 <= 0.625000000000000000E+00) { 
                        if (X1 <= 0.937500000000000000E-01) { 
                           return 105;
                        } else {
                           return 106;
                        }
                     } else {
                        if (X1 <= 0.937500000000000000E-01) { 
                           return 107;
                        } else {
                           return 108;
                        }
                     }
                  } else {
                     if (X1 <= 0.937500000000000000E-01) { 
                        return 109;
                     } else {
                        return 110;
                     }
                  }
               }
            } else {
               if (X2 <= 0.750000000000000000E+00) { 
                  if (X2 <= 0.625000000000000000E+00) { 
                     if (X1 <= 0.187500000000000000E+00) { 
                        if (X1 <= 0.156250000000000000E+00) { 
                           return 111;
                        } else {
                           return 112;
                        }
                     } else {
                        if (X1 <= 0.218750000000000000E+00) { 
                           return 113;
                        } else {
                           return 114;
                        }
                     }
                  } else {
                     if (X1 <= 0.187500000000000000E+00) { 
                        if (X1 <= 0.156250000000000000E+00) { 
                           return 115;
                        } else {
                           return 116;
                        }
                     } else {
                        if (X1 <= 0.218750000000000000E+00) { 
                           return 117;
                        } else {
                           return 118;
                        }
                     }
                  }
               } else {
                  if (X1 <= 0.187500000000000000E+00) { 
                     if (X2 <= 0.875000000000000000E+00) { 
                        return 119;
                     } else {
                        return 120;
                     }
                  } else {
                     if (X2 <= 0.875000000000000000E+00) { 
                        if (X1 <= 0.218750000000000000E+00) { 
                           return 121;
                        } else {
                           return 122;
                        }
                     } else {
                        return 123;
                     }
                  }
               }
            }
         } else {
            if (X1 <= 0.375000000000000000E+00) { 
               if (X2 <= 0.750000000000000000E+00) { 
                  if (X1 <= 0.312500000000000000E+00) { 
                     if (X2 <= 0.625000000000000000E+00) { 
                        if (X1 <= 0.281250000000000000E+00) { 
                           return 124;
                        } else {
                           return 125;
                        }
                     } else {
                        if (X1 <= 0.281250000000000000E+00) { 
                           return 126;
                        } else {
                           return 127;
                        }
                     }
                  } else {
                     if (X2 <= 0.625000000000000000E+00) { 
                        return 128;
                     } else {
                        if (X1 <= 0.343750000000000000E+00) { 
                           return 129;
                        } else {
                           return 130;
                        }
                     }
                  }
               } else {
                  if (X1 <= 0.312500000000000000E+00) { 
                     if (X2 <= 0.875000000000000000E+00) { 
                        if (X1 <= 0.281250000000000000E+00) { 
                           return 131;
                        } else {
                           return 132;
                        }
                     } else {
                        if (X1 <= 0.281250000000000000E+00) { 
                           return 133;
                        } else {
                           return 134;
                        }
                     }
                  } else {
                     if (X2 <= 0.875000000000000000E+00) { 
                        if (X1 <= 0.343750000000000000E+00) { 
                           return 135;
                        } else {
                           return 136;
                        }
                     } else {
                        return 137;
                     }
                  }
               }
            } else {
               if (X2 <= 0.750000000000000000E+00) { 
                  if (X1 <= 0.437500000000000000E+00) { 
                     if (X2 <= 0.625000000000000000E+00) { 
                        return 138;
                     } else {
                        return 139;
                     }
                  } else {
                     return 140;
                  }
               } else {
                  if (X1 <= 0.437500000000000000E+00) { 
                     if (X2 <= 0.875000000000000000E+00) { 
                        return 141;
                     } else {
                        return 142;
                     }
                  } else {
                     if (X2 <= 0.875000000000000000E+00) { 
                        return 143;
                     } else {
                        return 144;
                     }
						}
					}
				}
			}
		}
	} else {
		if (X1 <= 0.750000000000000000E+00) { 
			if (X2 <= 0.500000000000000000E+00) { 
				if (X1 <= 0.625000000000000000E+00) { 
					if (X2 <= 0.250000000000000000E+00) { 
						return 145;
					} else {
						return 146;
					}
				} else {
					return 147;
				}
			} else {
				if (X1 <= 0.625000000000000000E+00) { 
					if (X2 <= 0.750000000000000000E+00) { 
						if (X1 <= 0.562500000000000000E+00) { 
							return 148;
						} else {
							return 149;
						}
					} else {
						if (X1 <= 0.562500000000000000E+00) { 
							return 150;
						} else {
							return 151;
						}
					}
				} else {
					if (X1 <= 0.687500000000000000E+00) { 
						return 152;
					} else {
						return 153;
					}
				}
			}
		} else {
			return 154;
		}
	}
   return 255; // something is wrong
}
