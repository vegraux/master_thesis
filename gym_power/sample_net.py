# -*- coding: utf-8 -*-

"""

"""
import pandapower.networks as pn
import pandapower as pp
__author__ = 'Vegard Solberg'
__email__ = 'vegard.ulriksen.solberg@nmbu.no'


def simple_two_bus():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=20.)
    b2 = pp.create_bus(net, vn_kv=20.)
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=2.5,
                   std_type="NA2XS2Y 1x240 RM/25 12/20 kV")

    pp.create_ext_grid(net, bus=b2)
    pp.create_gen(net, bus=b1, p_kw=-1000, vn_kv=20, sn_kva=8000
                  , controllable=True)
    pp.runpp(net)
    return net


def cigre_network(with_der='pv_wind',scale_nominal_value=40):
    """
    Creates 15 bus cigre network with 8 solar units and 1 wind park.
    The network is scaled so that there is a lot of solar power production
    :param with_der:
    :return:
    """

    net = pn.create_cigre_network_mv(with_der="pv_wind")
    net.sgen['sn_kva'] *= scale_nominal_value
    net.sgen.loc[8, 'sn_kva'] /= scale_nominal_value # undo scaling for wind park
    return net

if __name__ == '__main__':
    simple_two_bus()