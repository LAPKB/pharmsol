pub(crate) const STRUCTURED_BLOCK_CORPUS: &str = r#"model one_cmt_oral_iv {
  kind ode
  parameters {
    ka,
    cl,
    v,
    tlag,
    f_oral,
  }
  covariates {
    wt @linear,
  }
  states {
    depot,
    central,
  }
  routes {
    oral -> depot {
      lag = tlag
      bioavailability = f_oral
    }

    iv -> central
  }
  derive {
    cl_i = cl * pow(wt / 70.0, 0.75)
    if wt > 120.0 {
      v_i = v * 1.15
    } else {
      v_i = v
    }
    ke = cl_i / v_i
  }
  dynamics {
    ddt(depot) = -ka * depot
    ddt(central) = ka * depot - ke * central
  }
  outputs {
    cp = central / v_i
  }
}


model transit_absorption {
  kind ode
  parameters {
    ktr,
    ke,
    v,
  }
  states {
    transit[4],
    central,
  }
  routes {
    oral -> transit[0]
  }
  dynamics {
    ddt(transit[0]) = -ktr * transit[0]
    for stage in 1..4 {
      ddt(transit[stage]) = ktr * transit[stage - 1] - ktr * transit[stage]
    }
    ddt(central) = ktr * transit[3] - ke * central
  }
  outputs {
    cp = central / v
  }
}


model one_cmt_abs {
  kind analytical
  parameters {
    ka,
    ke,
    v,
  }
  states {
    depot,
    central,
  }
  routes {
    oral -> depot
  }
  analytical {
    structure = one_compartment_with_absorption
  }
  outputs {
    cp = central / v
  }
}


model vanco_sde {
  kind sde
  parameters {
    ka,
    ke0,
    kcp,
    kpc,
    vol,
    ske,
  }
  covariates {
    wt @locf,
  }
  states {
    depot,
    central,
    peripheral,
    ke_latent,
  }
  routes {
    oral -> depot
  }
  particles 1000
  init {
    ke_latent = ke0
  }
  drift {
    ddt(depot) = -ka * depot
    ddt(central) = ka * depot - (ke_latent + kcp) * central + kpc * peripheral
    ddt(peripheral) = kcp * central - kpc * peripheral
    ddt(ke_latent) = -ke_latent + ke0
  }
  diffusion {
    noise(ke_latent) = ske
  }
  outputs {
    cp = central / (vol * wt)
  }
}
"#;
